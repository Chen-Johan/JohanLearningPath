from tensorflow import keras
import tensorflow as tf

# 检查 TensorFlow 是否识别到 GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


