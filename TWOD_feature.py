import numpy as np
from PIL import Image
from tensorflow.keras.applications.inception_v3 import preprocess_input

def extract_radiation_feature(img):
    gray = np.mean(img, axis=-1)
    return np.array(
        [np.mean(gray), np.std(gray), np.max(gray)],
        dtype=np.float32
    )

def extract_motion_feature(prev_img, curr_img):
    diff = np.abs(curr_img.astype(np.float32) - prev_img.astype(np.float32))
    return np.mean(diff, dtype=np.float32)

def extract_image_features(img_paths, ann_paths, backbone):
    img_feats_a, img_feats_b = [], []
    prev_img = None

    for img_path in img_paths:
        img = Image.open(img_path).resize((299, 299)).convert("RGB")
        img_np = np.array(img)

        img_input = preprocess_input(img_np[np.newaxis, ...])
        cnn_feat = backbone(img_input, training=False).numpy().squeeze()

        rad_feat = extract_radiation_feature(img_np)

        if prev_img is None:
            motion_feat = np.array([0.0], dtype=np.float32)
        else:
            motion_feat = np.array(
                [extract_motion_feature(prev_img, img_np)],
                dtype=np.float32
            )
        prev_img = img_np

        img_feats_a.append(np.concatenate([cnn_feat, rad_feat], axis=0))
        img_feats_b.append(np.concatenate([cnn_feat, motion_feat], axis=0))

    return (
        np.array(img_feats_a, dtype=np.float32),
        np.array(img_feats_b, dtype=np.float32)
    )
