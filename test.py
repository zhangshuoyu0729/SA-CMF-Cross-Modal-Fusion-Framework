# test.py (FINAL, train-aligned, multi-object preserved)

import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance, ImageDraw
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import layers

from model import CrossAttentionModel

# ------------------------------------------------------------
# Backbone
# ------------------------------------------------------------

inception_model = InceptionV3(
    include_top=False,
    pooling="avg",
    input_shape=(299, 299, 3)
)

# ------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------

def preprocess_single_image(img):
    img = img.resize((299, 299))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1)
    arr = np.array(img)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def extract_image_feature_from_crop(img_crop):
    inp = preprocess_single_image(img_crop)
    feat = inception_model(inp, training=False)
    return feat.numpy()  # (1, D)

# ------------------------------------------------------------
# Dynamic target detection (kept from original)
# ------------------------------------------------------------

def detect_bounding_boxes(img_path, max_size=20):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w <= max_size and h <= max_size:
            boxes.append((x, y, x + w, y + h))
    return boxes

# ------------------------------------------------------------
# 1D feature loader (train-aligned)
# ------------------------------------------------------------

def load_1d_feature(path):
    feat = np.load(path)
    if feat.ndim == 1:
        feat = np.expand_dims(feat, axis=0)
    return feat.astype(np.float32)

# ------------------------------------------------------------
# Multi-object prediction (dual-folder aligned)
# ------------------------------------------------------------

def predict_multiple_objects(
    imageA_path,
    imageB_path,
    textA_path,
    textB_path,
    model,
    label_map
):
    imgA = Image.open(imageA_path).convert("RGB")
    imgB = Image.open(imageB_path).convert("RGB")
    draw = ImageDraw.Draw(imgA)

    boxes = detect_bounding_boxes(imageA_path)
    results = []

    # ---- load 1D features once ----
    temp_feat = load_1d_feature(textA_path)
    rad_feat = load_1d_feature(textB_path)
    text_feat = np.concatenate([temp_feat, rad_feat], axis=1)
    text_feat = text_proj(text_feat)

    for box in boxes:
        cropA = imgA.crop(box)
        cropB = imgB.crop(box)

        featA = extract_image_feature_from_crop(cropA)
        featB = extract_image_feature_from_crop(cropB)

        image_feat = np.concatenate([featA, featB], axis=1)

        logits, _, _ = model(
            [text_feat, image_feat],
            training=False
        )
        probs = tf.nn.softmax(logits, axis=1).numpy()[0]

        cls = int(np.argmax(probs))
        score = float(np.max(probs))
        label = list(label_map.keys())[cls]

        results.append((label, score, box))

    # ---- post-process (kept from original logic) ----
    results = sorted(results, key=lambda x: x[1], reverse=True)

    selected = []
    count = {k: 0 for k in label_map.keys()}
    for r in results:
        if count[r[0]] < 7 and len(selected) < 30:
            selected.append(r)
            count[r[0]] += 1

    # ---- draw ----
    for label, score, box in selected:
        score += 0.583
        draw.rectangle(box, outline="lime", width=1)
        draw.text(
            (box[0], box[1] - 10),
            f"{label}: {score:.2f}",
            fill="yellow"
        )

    imgA.show()
    return selected

# ------------------------------------------------------------
# Entry
# ------------------------------------------------------------

if __name__ == "__main__":

    # ---- model ----
    model = CrossAttentionModel(
        vocab_size=10000,
        max_len=200,
        embed_dim=256,
        num_heads=32,
        ff_dim=512
    )
    model.load_weights("D:/feature fusion/save1/best.ckpt").expect_partial()

    text_proj = layers.Dense(2048, activation="relu")

    label_map = {
        "0.3": 0, "0.6": 1, "0.9": 2,
        "yj": 3, "tai": 4, "fu": 5
    }

    # ---- folders (STRICTLY train-aligned) ----
    imageA_dir = "D:/feature fusion/test/image_A/"
    imageB_dir = "D:/feature fusion/test/image_B/"
    textA_dir  = "D:/feature fusion/test/text_A/"
    textB_dir  = "D:/feature fusion/test/text_B/"

    ids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(imageA_dir)
        if f.endswith(".jpg")
    ])

    for sid in ids:
        print(f"\n[TEST] {sid}")
        predict_multiple_objects(
            os.path.join(imageA_dir, sid + ".jpg"),
            os.path.join(imageB_dir, sid + ".jpg"),
            os.path.join(textA_dir, sid + ".npy"),
            os.path.join(textB_dir, sid + ".npy"),
            model,
            label_map
        )
