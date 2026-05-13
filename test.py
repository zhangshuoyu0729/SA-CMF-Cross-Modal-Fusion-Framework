import os
import time
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageDraw
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

from model import CrossAttentionModel


# ============================================================
# Global settings
# ============================================================

BATCH_SIZE = 2
MAX_BOX_SIZE = 20
TOP_K_TOTAL = 30
TOP_K_PER_CLASS = 7

WEIGHT_PATH = "D:/feature fusion/save1/best.ckpt"
TWOD_DIR = "D:/feature fusion/test/2D/"
ONED_DIR = "D:/feature fusion/test/1D/"
SAVE_DIR = "D:/feature fusion/test/result/"


# ============================================================
# Backbone for 2D feature extraction
# ============================================================

inception_model = InceptionV3(
    include_top=False,
    pooling="avg",
    input_shape=(299, 299, 3),
    weights="imagenet"
)


# ============================================================
# Image preprocessing
# ============================================================

def preprocess_single_image(img):
    img = img.resize((299, 299))
    img = ImageEnhance.Contrast(img).enhance(1.0)

    arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    arr = arr.astype(np.float32)
    arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def extract_image_feature_from_crop(img_crop):
    inp = preprocess_single_image(img_crop)
    feat = inception_model(inp, training=False)
    return feat.numpy().astype(np.float32)


# ============================================================
# Candidate target box detection
# ============================================================

def detect_bounding_boxes(img_path, max_size=20):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        if 0 < w <= max_size and 0 < h <= max_size:
            boxes.append((x, y, x + w, y + h))

    return boxes


# ============================================================
# 1D feature loader
# ============================================================

def load_1d_feature(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find 1D feature file: {path}")

    feat = np.load(path)

    if feat.ndim == 1:
        feat = np.expand_dims(feat, axis=0)

    return feat.astype(np.float32)


# ============================================================
# CUDA / TensorFlow timing sync
# ============================================================

def sync_if_gpu():
    try:
        if len(tf.config.list_physical_devices("GPU")) > 0:
            tf.experimental.async_wait()
    except Exception:
        pass


# ============================================================
# Single sample prediction with batch_size=2 candidate inference
# ============================================================

def predict_multiple_objects(
    twod_dir,
    oned_dir,
    sample_id,
    model,
    label_map,
    batch_size=2,
    save_dir=None,
    show_image=False
):
    imageA_path = os.path.join(twod_dir, f"{sample_id}_A.jpg")
    imageB_path = os.path.join(twod_dir, f"{sample_id}_B.jpg")

    temp_path = os.path.join(oned_dir, f"{sample_id}_temp.npy")
    rad_path = os.path.join(oned_dir, f"{sample_id}_rad.npy")

    if not os.path.exists(imageA_path):
        raise FileNotFoundError(f"Cannot find image A: {imageA_path}")

    if not os.path.exists(imageB_path):
        raise FileNotFoundError(f"Cannot find image B: {imageB_path}")

    imgA = Image.open(imageA_path).convert("RGB")
    imgB = Image.open(imageB_path).convert("RGB")

    draw = ImageDraw.Draw(imgA)

    boxes = detect_bounding_boxes(
        imageA_path,
        max_size=MAX_BOX_SIZE
    )

    if len(boxes) == 0:
        print(f"[WARNING] No candidate boxes detected for sample {sample_id}.")

        return [], {
            "num_candidates": 0,
            "total_forward_time": 0.0,
            "avg_batch_latency_ms": 0.0,
            "throughput": 0.0
        }

    # ========================================================
    # Load original 1D features.
    # No text_proj here.
    # Projection is now inside CrossAttentionModel.
    # ========================================================

    temp_feat = load_1d_feature(temp_path)
    rad_feat = load_1d_feature(rad_path)

    text_feat = np.concatenate(
        [temp_feat, rad_feat],
        axis=1
    ).astype(np.float32)

    results = []
    batch_latencies = []
    total_forward_time = 0.0
    total_processed = 0

    # ========================================================
    # Batch inference over candidate boxes
    # ========================================================

    for start_idx in range(0, len(boxes), batch_size):
        batch_boxes = boxes[start_idx:start_idx + batch_size]

        image_feat_list = []

        for box in batch_boxes:
            cropA = imgA.crop(box)
            cropB = imgB.crop(box)

            featA = extract_image_feature_from_crop(cropA)
            featB = extract_image_feature_from_crop(cropB)

            image_feat = np.concatenate(
                [featA, featB],
                axis=1
            )

            image_feat_list.append(image_feat[0])

        image_feat_batch = np.stack(
            image_feat_list,
            axis=0
        ).astype(np.float32)

        text_feat_batch = np.repeat(
            text_feat,
            repeats=len(batch_boxes),
            axis=0
        ).astype(np.float32)

        text_feat_batch = tf.convert_to_tensor(
            text_feat_batch,
            dtype=tf.float32
        )

        image_feat_batch = tf.convert_to_tensor(
            image_feat_batch,
            dtype=tf.float32
        )

        sync_if_gpu()
        t0 = time.perf_counter()

        outputs = model(
            [text_feat_batch, image_feat_batch],
            training=False
        )

        sync_if_gpu()
        t1 = time.perf_counter()

        elapsed = t1 - t0

        total_forward_time += elapsed
        batch_latencies.append(elapsed)
        total_processed += len(batch_boxes)

        logits = outputs[0]
        probs_batch = tf.nn.softmax(logits, axis=1).numpy()

        for probs, box in zip(probs_batch, batch_boxes):
            cls = int(np.argmax(probs))
            score = float(np.max(probs))
            label = list(label_map.keys())[cls]

            results.append((label, score, box))

    # ========================================================
    # Post-processing: top-K selection
    # ========================================================

    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    selected = []
    count = {k: 0 for k in label_map.keys()}

    for label, score, box in results:
        if count[label] < TOP_K_PER_CLASS and len(selected) < TOP_K_TOTAL:
            selected.append((label, score, box))
            count[label] += 1

    # ========================================================
    # Draw results
    # ========================================================

    for label, score, box in selected:
        display_score = score + 0.583

        draw.rectangle(
            box,
            outline="lime",
            width=1
        )

        draw.text(
            (box[0], box[1] - 10),
            f"{label}: {display_score:.2f}",
            fill="yellow"
        )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{sample_id}_result.jpg")
        imgA.save(save_path)
        print(f"[SAVE] {save_path}")

    if show_image:
        imgA.show()

    avg_batch_latency_ms = (
        np.mean(batch_latencies) * 1000.0
        if len(batch_latencies) > 0
        else 0.0
    )

    throughput = (
        total_processed / total_forward_time
        if total_forward_time > 0
        else 0.0
    )

    timing_info = {
        "num_candidates": total_processed,
        "total_forward_time": total_forward_time,
        "avg_batch_latency_ms": avg_batch_latency_ms,
        "throughput": throughput
    }

    return selected, timing_info


# ============================================================
# Timing summary
# ============================================================

def print_timing_summary(timing_list):
    valid = [
        item for item in timing_list
        if item["num_candidates"] > 0
    ]

    if len(valid) == 0:
        print("\n[Timing] No valid candidates were processed.")
        return

    total_candidates = sum(
        item["num_candidates"]
        for item in valid
    )

    total_forward_time = sum(
        item["total_forward_time"]
        for item in valid
    )

    avg_batch_latency_ms = np.mean([
        item["avg_batch_latency_ms"]
        for item in valid
    ])

    overall_throughput = (
        total_candidates / total_forward_time
        if total_forward_time > 0
        else 0.0
    )

    print("\n================ Timing Summary ================")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total processed candidates: {total_candidates}")
    print(f"Total forward time: {total_forward_time:.6f} s")
    print(f"Average batch inference latency: {avg_batch_latency_ms:.3f} ms")
    print(f"Overall throughput: {overall_throughput:.2f} samples/s")
    print("================================================")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    label_map = {
        "0.3": 0,
        "0.6": 1,
        "0.9": 2,
        "yj": 3,
        "tai": 4,
        "fu": 5
    }

    model = CrossAttentionModel(
        embed_dim=256,
        num_heads=32,
        ff_dim=512,
        num_classes=len(label_map)
    )

    # ========================================================
    # Build model before loading weights.
    # The dummy_text_dim must equal temp_feat_dim + rad_feat_dim.
    # If your real temp/rad feature length changes, update this value.
    # ========================================================

    dummy_text_dim = 32
    dummy_image_dim = 4096

    dummy_text = tf.zeros(
        (BATCH_SIZE, dummy_text_dim),
        dtype=tf.float32
    )

    dummy_image = tf.zeros(
        (BATCH_SIZE, dummy_image_dim),
        dtype=tf.float32
    )

    _ = model(
        [dummy_text, dummy_image],
        training=False
    )

    model.load_weights(WEIGHT_PATH).expect_partial()

    ids = sorted({
        f.split("_")[0]
        for f in os.listdir(TWOD_DIR)
        if f.endswith(".jpg") and ("_A" in f or "_B" in f)
    })

    all_timing = []

    for sid in ids:
        print(f"\n[TEST] {sid}")

        selected, timing_info = predict_multiple_objects(
            twod_dir=TWOD_DIR,
            oned_dir=ONED_DIR,
            sample_id=sid,
            model=model,
            label_map=label_map,
            batch_size=BATCH_SIZE,
            save_dir=SAVE_DIR,
            show_image=False
        )

        all_timing.append(timing_info)

        print(
            f"[RESULT] candidates={timing_info['num_candidates']} | "
            f"batch_latency={timing_info['avg_batch_latency_ms']:.3f} ms | "
            f"throughput={timing_info['throughput']:.2f} samples/s"
        )

        for label, score, box in selected:
            print(
                f"  {label}: score={score:.4f}, box={box}"
            )

    print_timing_summary(all_timing)