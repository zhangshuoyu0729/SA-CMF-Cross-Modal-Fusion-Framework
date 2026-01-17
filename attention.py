import tensorflow as tf

def calculate_attention_weights(feature_matrix, importance_vector):
    # 计算注意力权重
    attention_scores = tf.nn.softmax(tf.matmul(feature_matrix, importance_vector), axis=-1)
    return attention_scores


# 结合注意力权重计算最终的图像和文本特征
def apply_attention(feature_matrix, attention_weights):
    weighted_features = feature_matrix * attention_weights
    final_feature = tf.reduce_sum(weighted_features, axis=0)
    return final_feature
