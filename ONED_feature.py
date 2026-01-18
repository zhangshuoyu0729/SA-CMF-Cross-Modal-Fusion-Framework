import tensorflow as tf
import numpy as np
import os

def load_text_features(save_dir, total_samples):
    feats = []
    for i in range(total_samples):
        path = os.path.join(save_dir, f"{i}.npy")
        feats.append(np.load(path))
    return np.vstack(feats)

def load_text_features_as_tensor(save_dir, total_samples, split_idx):
    raw = load_text_features(save_dir, total_samples)

    temp_feat = raw[:, :split_idx]
    rad_feat = raw[:, split_idx:]

    return (
        tf.convert_to_tensor(temp_feat, dtype=tf.float32),
        tf.convert_to_tensor(rad_feat, dtype=tf.float32)
    )
