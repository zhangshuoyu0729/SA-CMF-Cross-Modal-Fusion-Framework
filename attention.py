import tensorflow as tf

def calculate_attention_weights(feature_matrix, importance_vector):
    # Calculate attention weights
    attention_scores = tf.nn.softmax(tf.matmul(feature_matrix, importance_vector), axis=-1)
    return attention_scores


# The final image and text features are calculated by combining attention weights.
def apply_attention(feature_matrix, attention_weights):
    weighted_features = feature_matrix * attention_weights
    final_feature = tf.reduce_sum(weighted_features, axis=0)
    return final_feature
