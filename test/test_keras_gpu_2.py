import tensorflow as tf
from tensorflow.keras import layers, models

# 检查是否有GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 构建一个简单的卷积神经网络
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 生成一些虚拟数据来训练模型
import numpy as np
train_images = np.random.random((1000, 28, 28, 1))
train_labels = np.random.randint(10, size=(1000,))

# 训练模型（默认使用 GPU 加速）
model.fit(train_images, train_labels, epochs=10)
