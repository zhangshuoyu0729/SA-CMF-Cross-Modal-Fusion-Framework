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
                 num_layers=5,
                 num_classes=6):
        super().__init__()

        self.backbone = InceptionV3(
            include_top=False,
            pooling="avg",
            weights="imagenet"
        )

        # 新增：1D特征投影层，参与训练和保存
        self.text_proj = tf.keras.layers.Dense(
            2048,
            activation="relu",
            name="text_proj"
        )

        self.img_proj = tf.keras.layers.Dense(embed_dim)
        self.seq_proj = tf.keras.layers.Dense(embed_dim)

        self.transformer_layers = [
            TransformerEncoder(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

        self.cross_attn = CrossAttention(embed_dim)

        self.alpha = 0.3
        self.beta = 0.7

        self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, inputs, training=False):
        """
        inputs:
            text_feat: [B, D1]
            image_feat: [B, D2]
        """

        text_feat, image_feat = inputs

        # text_feat 内部投影，权重随模型保存
        text_feat = self.text_proj(text_feat)

        # 拆分一维特征
        text_mid = text_feat.shape[-1] // 2
        seq_a = text_feat[:, :text_mid]
        seq_b = text_feat[:, text_mid:]

        # 拆分二维图像特征
        img_mid = image_feat.shape[-1] // 2
        img_rad = image_feat[:, :img_mid]
        img_motion = image_feat[:, img_mid:]

        # 加 sequence 维度
        seq_a = tf.expand_dims(seq_a, axis=1)
        seq_b = tf.expand_dims(seq_b, axis=1)

        img_feat = self.img_proj(img_rad + img_motion)
        img_feat = tf.expand_dims(img_feat, axis=1)

        seq_feat = self.seq_proj(seq_a + seq_b)

        for layer in self.transformer_layers:
            seq_feat = layer(seq_feat)

        cross_feat, Atten_cro = self.cross_attn(img_feat, seq_feat)

        mi_map = mutual_information_map(img_feat, seq_feat)

        weight_img = tf.ones_like(img_feat[..., :1])
        weight_seq = tf.ones_like(seq_feat[..., :1])

        Im_img_p, Im_seq_p = update_importance(
            mi_map, weight_img, weight_seq
        )

        img_att = tf.nn.softmax(Im_img_p, axis=1) * img_feat
        seq_att = tf.nn.softmax(Im_seq_p, axis=1) * seq_feat

        Final_Atten = (
            self.alpha * cross_feat +
            self.beta * (img_att + seq_att)
        )

        Final_img = tf.reduce_mean(img_att, axis=1)
        Final_seq = tf.reduce_mean(seq_att, axis=1)

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