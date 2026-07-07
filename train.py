import tensorflow as tf
import numpy as np
import json
import os
import time

from model import CrossAttentionModel
from TWOD_feature import extract_image_features
from ONED_feature import load_text_features_as_tensor
from metrics import (
    expected_calibration_error,
    feature_entropy,
    l2_normalize_np,
    mcg_components_from_fusion_chain,
    missing_modality_evaluation,
    modality_complementarity_gain,
    multiclass_auc,
    multiclass_metrics,
    recall_at_k,
    redundancy_score,
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


def dual_triplet_loss(F_img, F_txt, margin=0.3):
    """
    Implements with batch-wise negatives
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
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_classes=6
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
            Final_img, Final_text, Final_Atten
        )

    # -------------------------------
    # Epoch loop
    # -------------------------------
    for epoch in range(epochs):

        meter = {
            "total": [], "cls": [], "sim": [], "mi": []
        }

        all_logits, all_labels = [], []
        all_missing_1d_logits, all_missing_2d_logits = [], []
        all_img_feat, all_txt_feat, all_fused_feat = [], [], []
        all_missing_1d_fused_feat, all_missing_2d_fused_feat = [], []

        start = time.time()

        for tb, ib, yb in dataset:
            out = train_step(tb, ib, yb)
            logits, tl, cl, sl, ml, fi, ft, ff = out

            meter["total"].append(tl.numpy())
            meter["cls"].append(cl.numpy())
            meter["sim"].append(sl.numpy())
            meter["mi"].append(ml.numpy())

            clean_eval = model([tb, ib], training=False)
            missing_1d_eval = model([tf.zeros_like(tb), ib], training=False)
            missing_2d_eval = model([tb, tf.zeros_like(ib)], training=False)

            all_logits.append(clean_eval[0].numpy())
            all_missing_1d_logits.append(missing_1d_eval[0].numpy())
            all_missing_2d_logits.append(missing_2d_eval[0].numpy())
            all_labels.append(yb.numpy())
            all_img_feat.append(clean_eval[1].numpy())
            all_txt_feat.append(clean_eval[2].numpy())
            all_fused_feat.append(tf.reduce_mean(clean_eval[3], axis=1).numpy())
            all_missing_1d_fused_feat.append(
                tf.reduce_mean(missing_1d_eval[3], axis=1).numpy()
            )
            all_missing_2d_fused_feat.append(
                tf.reduce_mean(missing_2d_eval[3], axis=1).numpy()
            )

        all_logits = np.vstack(all_logits)
        all_missing_1d_logits = np.vstack(all_missing_1d_logits)
        all_missing_2d_logits = np.vstack(all_missing_2d_logits)
        all_labels = np.concatenate(all_labels)
        probs = tf.nn.softmax(all_logits).numpy()
        missing_1d_probs = tf.nn.softmax(all_missing_1d_logits).numpy()
        missing_2d_probs = tf.nn.softmax(all_missing_2d_logits).numpy()
        preds = np.argmax(probs, axis=1)

        img_feat = np.vstack(all_img_feat)
        txt_feat = np.vstack(all_txt_feat)
        fused_feat = np.vstack(all_fused_feat)
        missing_1d_fused_feat = np.vstack(all_missing_1d_fused_feat)
        missing_2d_fused_feat = np.vstack(all_missing_2d_fused_feat)

        # -------------------------------
        # Metrics
        # -------------------------------
        acc = np.mean(preds == all_labels)
        cls_metrics = multiclass_metrics(all_labels, preds)
        cls_prec = cls_metrics["precision_weighted"]
        cls_rec = cls_metrics["recall_weighted"]
        cls_f1 = cls_metrics["f1_weighted"]
        roc_auc = multiclass_auc(
            all_labels, probs, num_classes=probs.shape[1]
        )

        sim_matrix = l2_normalize_np(img_feat) @ l2_normalize_np(txt_feat).T
        r1 = recall_at_k(sim_matrix, all_labels, 1)
        r5 = recall_at_k(sim_matrix, all_labels, 5)
        r10 = recall_at_k(sim_matrix, all_labels, 10)

        entropy = feature_entropy(fused_feat)
        rs = redundancy_score(sim_matrix)
        mcg = modality_complementarity_gain(img_feat, txt_feat, fused_feat)
        mcg_chain = mcg_components_from_fusion_chain(img_feat, txt_feat, fused_feat)
        rmm_1d_eval = missing_modality_evaluation(
            probs, missing_1d_probs, fused_feat,
            missing_1d_fused_feat, all_labels
        )
        rmm_2d_eval = missing_modality_evaluation(
            probs, missing_2d_probs, fused_feat,
            missing_2d_fused_feat, all_labels
        )
        ece = expected_calibration_error(probs, all_labels)

        print(
            f"\n[Epoch {epoch+1}] "
            f"Acc={acc:.4f} | Cls_Prec_w={cls_prec:.4f} | "
            f"Cls_Recall_w={cls_rec:.4f} | Cls_F1_w={cls_f1:.4f} | "
            f"AUC_wOvR={roc_auc:.4f}\n"
            f" Cls_Micro: Prec={cls_metrics['precision_micro']:.4f} | "
            f"Recall={cls_metrics['recall_micro']:.4f} | "
            f"F1={cls_metrics['f1_micro']:.4f}\n"
            f" Cls_Macro: Prec={cls_metrics['precision_macro']:.4f} | "
            f"Recall={cls_metrics['recall_macro']:.4f} | "
            f"F1={cls_metrics['f1_macro']:.4f}\n"
            f" Cat_R@1={r1:.4f} | Cat_R@5={r5:.4f} | Cat_R@10={r10:.4f}\n"
            f" Entropy={entropy:.4f} | RS={rs:.4f} | MCG={mcg:.4f} | "
            f"MCG_fusion={mcg_chain['fusion_gain']:.4f} | "
            f"MCG_redundancy={mcg_chain['redundancy_baseline']:.4f}\n"
            f" R@MM_1D={rmm_1d_eval['rmm']:.4f} "
            f"(AccRet={rmm_1d_eval['accuracy_retention']:.4f}, "
            f"ConfRet={rmm_1d_eval['confidence_retention']:.4f}, "
            f"FeatStab={rmm_1d_eval['feature_stability']:.4f}) | "
            f"R@MM_2D={rmm_2d_eval['rmm']:.4f} "
            f"(AccRet={rmm_2d_eval['accuracy_retention']:.4f}, "
            f"ConfRet={rmm_2d_eval['confidence_retention']:.4f}, "
            f"FeatStab={rmm_2d_eval['feature_stability']:.4f}) | "
            f"ECE={ece:.4f}\n"
            f" Loss: Total={np.mean(meter['total']):.4f} | "
            f"Cls={np.mean(meter['cls']):.4f} | "
            f"Sim={np.mean(meter['sim']):.4f} | "
            f"MI={np.mean(meter['mi']):.4f} | "
            f"Time={time.time()-start:.1f}s"
        )

        if cls_f1 > best_f1:
            best_f1 = cls_f1
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
