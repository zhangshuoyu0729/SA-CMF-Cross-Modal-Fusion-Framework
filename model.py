# model.py
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from fusion import (
    CrossAttention,
    mutual_information_map,
    update_importance
)

# =====================================================
# Transformer Encoder
# =====================================================
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, ff_dim):
        super().__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads, dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(dim)
        ])
        self.norm1 = tf.keras.layers.LayerNormalization()
        self.norm2 = tf.keras.layers.LayerNormalization()

    def call(self, x):
        attn_out = self.attn(x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


# =====================================================
# Full Model (MI-driven)
# =====================================================
class CrossAttentionModel(tf.keras.Model):
    def __init__(self,
                 embed_dim=256,
                 num_heads=8,
                 ff_dim=512,
                 num_classes=6):
        super().__init__()

        # -------- Backbones --------
        self.backbone = InceptionV3(
            include_top=False, pooling="avg", weights="imagenet"
        )

        self.img_proj = tf.keras.layers.Dense(embed_dim)
        self.seq_proj = tf.keras.layers.Dense(embed_dim)

        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim)

        self.cross_attn = CrossAttention(embed_dim)

        # Eq.(25) hyper-parameters
        self.alpha = 0.3
        self.beta = 0.7

        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs):
        """
        Inputs:
            (seq_a, seq_b), (img_rad, img_motion)

        Outputs:
            logits
            Final_img      -> Eq.(26)
            Final_seq      -> Eq.(26)
            Final_Atten    -> analysis / visualization
            Im_img_p
            Im_seq_p
        """
        (seq_a, seq_b), (img_rad, img_motion) = inputs

        # =================================================
        # 1. Feature projection
        # =================================================
        img_feat = self.img_proj(img_rad + img_motion)   # [B, D]
        img_feat = tf.expand_dims(img_feat, axis=1)      # [B, Ti=1, D]

        seq_feat = self.seq_proj(seq_a + seq_b)
        seq_feat = self.transformer(seq_feat)            # [B, To, D]

        # =================================================
        # 2. Cross-Attention
        # =================================================
        cross_feat, Atten_cro = self.cross_attn(img_feat, seq_feat)
        # cross_feat: [B, Ti, D]

        # =================================================
        # 3. Mutual-Information-driven importance
        # =================================================
        mi_map = mutual_information_map(img_feat, seq_feat)

        weight_img = tf.ones_like(img_feat[..., :1])  # weight_img(ti,tj)
        weight_seq = tf.ones_like(seq_feat[..., :1])  # weight_onedim(oi)

        Im_img_p, Im_seq_p = update_importance(
            mi_map, weight_img, weight_seq
        )

        # =================================================
        # 4. Importance-weighted modality features
        # =================================================
        img_att = tf.nn.softmax(Im_img_p, axis=1) * img_feat
        seq_att = tf.nn.softmax(Im_seq_p, axis=1) * seq_feat

        # =================================================
        # 5. Final_Attention
        # =================================================
        Final_Atten = (
            self.alpha * cross_feat +
            self.beta * (img_att + seq_att)
        )

        # =================================================
        # 6. Outputs for Triplet Loss
        # =================================================
        Final_img = tf.reduce_mean(img_att, axis=1)   # Final_img(x)
        Final_seq = tf.reduce_mean(seq_att, axis=1)   # Final_onedim(y')

        fused = tf.reduce_mean(Final_Atten, axis=1)
        logits = self.classifier(fused)

        return (
            logits,
            Final_img,
            Final_seq,
            Final_Atten,
            Im_img_p,
            Im_seq_p
        )
