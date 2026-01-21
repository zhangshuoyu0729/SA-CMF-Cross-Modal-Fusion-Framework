import tensorflow as tf
import numpy as np
import json
import os
import time

from model import CrossAttentionModel
from TWOD_feature import extract_image_features
from ONED_feature import load_text_features_as_tensor

from tensorflow.keras import layers
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_curve, auc
)

tf.config.run_functions_eagerly(True)

# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------

def extract_labels_from_annotation(annotation_paths):
    label_map = {'0.3': 0, '0.6': 1, '0.9': 2, 'yj': 3, 'tai': 4, 'fu': 5}
    labels = []
    for p in annotation_paths:
        with open(p, 'r') as f:
            ann = json.load(f)
        labels.append(label_map[ann['shapes'][0]['label']])
    return np.array(labels, dtype=np.int32)


def cosine_sim(a, b):
    a = tf.nn.l2_normalize(a, axis=1)
    b = tf.nn.l2_normalize(b, axis=1)
    return tf.reduce_sum(a * b, axis=1)


def entropy_of_feature(f):
    p = tf.nn.softmax(f, axis=1)
    return -tf.reduce_mean(tf.reduce_sum(p * tf.math.log(p + 1e-9), axis=1))


def expected_calibration_error(probs, labels, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if np.any(mask):
            acc = np.mean(accuracies[mask])
            conf = np.mean(confidences[mask])
            ece += np.abs(acc - conf) * np.sum(mask) / len(labels)
    return ece


def recall_at_k(sim_matrix, labels, k):
    correct = 0
    for i in range(sim_matrix.shape[0]):
        topk = np.argsort(sim_matrix[i])[::-1][:k]
        if labels[i] in labels[topk]:
            correct += 1
    return correct / len(labels)


def dual_triplet_loss(F_img, F_txt, margin=0.3):
    """
    Implements Eq.(26)-(27) with batch-wise negatives
    F_img: [B, D]  -> Final_img
    F_txt: [B, D]  -> Final_text
    """
    # cosine similarity matrix
    F_img = tf.nn.l2_normalize(F_img, axis=1)
    F_txt = tf.nn.l2_normalize(F_txt, axis=1)
    sim = tf.matmul(F_img, F_txt, transpose_b=True)  # [B, B]

    # positive similarity S(x, y')
    pos = tf.linalg.diag_part(sim)  # [B]

    # negative similarities
    large_neg = tf.eye(tf.shape(sim)[0]) * 1e9
    neg_img = tf.reduce_max(sim - large_neg, axis=1)  # S(x̂, y')
    neg_txt = tf.reduce_max(sim - large_neg, axis=0)  # S(x, ŷ')

    loss_img = tf.maximum(0.0, margin - pos + neg_img)
    loss_txt = tf.maximum(0.0, margin - pos + neg_txt)

    return tf.reduce_mean(loss_img + loss_txt)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

def train_model(vocab_size, max_len, embed_dim,
                num_heads, ff_dim,
                batch_size, epochs):

    save_dir = "D:/feature fusion/save1/"
    model_dir = "D:/feature fusion/save/"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    model = CrossAttentionModel(
        vocab_size, max_len, embed_dim, num_heads, ff_dim
    )
    optimizer = tf.keras.optimizers.Adam(1e-3)

    TWOD_folder = "D:/feature fusion/data/2D/"
    ONED_folder = "D:/feature fusion/data/1D/"

    image_paths = sorted([
        os.path.join(TWOD_folder, f)
        for f in os.listdir(TWOD_folder) if f.endswith(".jpg")
    ])
    annotation_paths = [
        p.replace(".jpg", ".json") for p in image_paths
    ]

    # -------------------------------
    # Load features
    # -------------------------------
    img_a, img_b = extract_image_features(
        image_paths, annotation_paths, model.backbone
    )
    image_features = np.concatenate([img_a, img_b], axis=1)

    temp_feat, rad_feat = load_text_features_as_tensor(
        ONED_folder, len(image_paths), split_idx=16
    )
    text_features = tf.concat([temp_feat, rad_feat], axis=1)
    text_features = layers.Dense(2048, activation='relu')(text_features)

    labels = extract_labels_from_annotation(annotation_paths)

    dataset = tf.data.Dataset.from_tensor_slices(
        (text_features, image_features, labels)
    ).shuffle(len(labels)).batch(batch_size)

    best_f1 = 0.0

    # -------------------------------
    # Train step
    # -------------------------------
    @tf.function
    def train_step(text_b, img_b, y_b):
        with tf.GradientTape() as tape:

            # <<< CHANGED: model outputs >>>
            logits, Final_img, Final_text, Final_Atten, Im_img_p, Im_text_p = \
                model([text_b, img_b])

            cls_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True)(y_b, logits)

            triplet_loss = dual_triplet_loss(Final_img, Final_text, margin=0.3)

            # MI regularization (not replacing fusion)
            mi_loss = tf.reduce_mean(Im_img_p) + tf.reduce_mean(Im_text_p)

            total_loss = cls_loss + 0.01 * triplet_loss + 0.01 * mi_loss

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return (
            logits, total_loss, cls_loss,
            triplet_loss, mi_loss,
            Final_img, Final_text
        )

    # -------------------------------
    # Epoch loop
    # -------------------------------
    for epoch in range(epochs):

        meter = {
            "total": [], "cls": [], "sim": [], "mi": []
        }

        all_logits, all_labels = [], []
        all_img_feat, all_txt_feat = [], []

        start = time.time()

        for tb, ib, yb in dataset:
            out = train_step(tb, ib, yb)
            logits, tl, cl, sl, ml, fi, ft = out

            meter["total"].append(tl.numpy())
            meter["cls"].append(cl.numpy())
            meter["sim"].append(sl.numpy())
            meter["mi"].append(ml.numpy())

            all_logits.append(logits.numpy())
            all_labels.append(yb.numpy())
            all_img_feat.append(fi.numpy())
            all_txt_feat.append(ft.numpy())

        all_logits = np.vstack(all_logits)
        all_labels = np.concatenate(all_labels)
        probs = tf.nn.softmax(all_logits).numpy()
        preds = np.argmax(probs, axis=1)

        img_feat = np.vstack(all_img_feat)
        txt_feat = np.vstack(all_txt_feat)

        # -------------------------------
        # Metrics
        # -------------------------------
        acc = np.mean(preds == all_labels)
        f1 = f1_score(all_labels, preds, average="macro")
        prec = precision_score(all_labels, preds, average="macro")
        rec = recall_score(all_labels, preds, average="macro")

        fpr, tpr, _ = roc_curve(all_labels, probs[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)

        sim_matrix = img_feat @ txt_feat.T
        r1 = recall_at_k(sim_matrix, all_labels, 1)
        r5 = recall_at_k(sim_matrix, all_labels, 5)
        r10 = recall_at_k(sim_matrix, all_labels, 10)

        entropy = entropy_of_feature(img_feat).numpy()
        rs = np.mean(np.std(sim_matrix, axis=1))
        mcg = np.mean(np.max(sim_matrix, axis=1))
        ece = expected_calibration_error(probs, all_labels)

        print(
            f"\n[Epoch {epoch+1}] "
            f"Acc={acc:.4f} | Prec={prec:.4f} | Recall={rec:.4f} | "
            f"F1={f1:.4f} | AUC={roc_auc:.4f}\n"
            f" R@1={r1:.4f} | R@5={r5:.4f} | R@10={r10:.4f}\n"
            f" Entropy={entropy:.4f} | RS={rs:.4f} | MCG={mcg:.4f} | "
            f"ECE={ece:.4f}\n"
            f" Loss: Total={np.mean(meter['total']):.4f} | "
            f"Cls={np.mean(meter['cls']):.4f} | "
            f"Sim={np.mean(meter['sim']):.4f} | "
            f"MI={np.mean(meter['mi']):.4f} | "
            f"Time={time.time()-start:.1f}s"
        )

        if f1 > best_f1:
            best_f1 = f1
            model.save_weights(os.path.join(save_dir, "best.ckpt"))
            model.save(os.path.join(model_dir, "best_model"))


# ------------------------------------------------------------------
if __name__ == "__main__":
    train_model(
        vocab_size=10000,
        max_len=200,
        embed_dim=256,
        num_heads=32,
        ff_dim=512,
        batch_size=16,
        epochs=100
    )
