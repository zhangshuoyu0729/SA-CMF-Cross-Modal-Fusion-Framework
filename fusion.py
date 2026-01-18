# fusion.py
import tensorflow as tf
import math

# =====================================================
# Cross-Attention Module
# =====================================================
class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.q = tf.keras.layers.Dense(dim)
        self.k = tf.keras.layers.Dense(dim)
        self.v = tf.keras.layers.Dense(dim)
        self.scale = math.sqrt(dim)

    def call(self, q_in, kv_in):
        Q = self.q(q_in)
        K = self.k(kv_in)
        V = self.v(kv_in)

        attn = tf.nn.softmax(
            tf.matmul(Q, K, transpose_b=True) / self.scale,
            axis=-1
        )
        return tf.matmul(attn, V), attn


# =====================================================
# ECDF Mapping
# =====================================================
def ecdf(x):
    """
    x: [..., N]
    """
    x_sorted = tf.sort(x, axis=-1)
    ranks = tf.argsort(tf.argsort(x, axis=-1), axis=-1)
    n = tf.cast(tf.shape(x)[-1], tf.float32)
    return (tf.cast(ranks, tf.float32) + 1.0) / (n + 1.0)


# =====================================================
# Gumbel Copula
# =====================================================
def gumbel_copula(u, v, theta=2.0):
    eps = 1e-6
    u = tf.clip_by_value(u, eps, 1.0 - eps)
    v = tf.clip_by_value(v, eps, 1.0 - eps)

    inner = tf.pow(-tf.math.log(u), theta) + tf.pow(-tf.math.log(v), theta)
    return tf.exp(-tf.pow(inner, 1.0 / theta))


# =====================================================
# Mutual Information Estimation
# =====================================================
def mutual_information_map(img_feat, seq_feat, theta=2.0):
    """
    img_feat: [B, Ti, D]
    seq_feat: [B, To, D]
    return:
        MI_map: [B, Ti, To]
    """
    # project to scalar per position
    img_score = tf.reduce_mean(img_feat, axis=-1)
    seq_score = tf.reduce_mean(seq_feat, axis=-1)

    u = ecdf(img_score)
    v = ecdf(seq_score)

    u = tf.expand_dims(u, axis=-1)
    v = tf.expand_dims(v, axis=1)

    c = gumbel_copula(u, v, theta)

    # approximate MI
    mi = c * tf.math.log(c + 1e-6)
    return mi


# =====================================================
# Importance Update
# =====================================================
def update_importance(mi_map, weight_img, weight_seq):
    """
    mi_map: [B, Ti, To]
    weight_img: [B, Ti, 1]
    weight_seq: [B, To, 1]
    """
    mi_sum = tf.reduce_sum(mi_map, axis=[1, 2], keepdims=True)

    Im_img = (
        weight_img *
        tf.reduce_sum(mi_map, axis=2, keepdims=True)
    ) / (mi_sum + 1e-6)

    Im_seq = (
        weight_seq *
        tf.reduce_sum(mi_map, axis=1, keepdims=True)
    ) / (mi_sum + 1e-6)

    return Im_img, Im_seq
