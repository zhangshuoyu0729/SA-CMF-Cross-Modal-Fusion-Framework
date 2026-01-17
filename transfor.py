import tensorflow as tf

# 特征转换模块
def feature_transformation(input_features, num_groups, stride):
    transformed_features = []
    for i in range(num_groups):
        conv = tf.keras.layers.Conv1D(1, 1, strides=stride)(input_features[:, i:i+1, :])
        transformed_features.append(conv)
    return tf.concat(transformed_features, axis=1)


# 保存转换后的特征
def save_transformed_features(features, save_path):
    features_np = features.numpy()  # 将转换后的特征转换为 NumPy 数组
    tf.io.write_file(save_path, tf.io.serialize_tensor(features_np))

