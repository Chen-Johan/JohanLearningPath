# -------------------------------------------------------------#
#   MobileNet的网络部分（TensorFlow 2版本）
# -------------------------------------------------------------#
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import DepthwiseConv2D, Input, Activation, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras import backend as K

def relu6(x):
    return tf.nn.relu6(x)

def MobileNet(input_shape=[224,224,3],
              depth_multiplier=1,
              dropout=1e-3,
              classes=1000):

    img_input = Input(shape=input_shape)

    # 224,224,3 -> 112,112,32
    x = _conv_block(img_input, 32, strides=(2, 2))

    # 112,112,32 -> 112,112,64
    x = _depthwise_conv_block(x, 64, depth_multiplier, block_id=1)

    # 112,112,64 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier,
                              strides=(2, 2), block_id=2)
    # 56,56,128 -> 56,56,128
    x = _depthwise_conv_block(x, 128, depth_multiplier, block_id=3)

    # 56,56,128 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier,
                              strides=(2, 2), block_id=4)
    
    # 28,28,256 -> 28,28,256
    x = _depthwise_conv_block(x, 256, depth_multiplier, block_id=5)

    # 28,28,256 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier,
                              strides=(2, 2), block_id=6)
    
    # 14,14,512 -> 14,14,512
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, depth_multiplier, block_id=11)

    # 14,14,512 -> 7,7,1024
    x = _depthwise_conv_block(x, 1024, depth_multiplier,
                              strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, depth_multiplier, block_id=13)

    # 7,7,1024 -> 1,1,1024
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1024), name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_1_0_224_tf')
    return model

def _conv_block(inputs, filters, kernel=(3, 3), strides=(1, 1)):
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def _depthwise_conv_block(inputs, pointwise_conv_filters,
                          depth_multiplier=1, strides=(1, 1), block_id=1):

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)

    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)

if __name__ == '__main__':
    # 创建MobileNet模型
    model = MobileNet()

    # 输出模型结构
    model.summary()

    # 加载预训练权重
    weights_path = 'C:/Users/Gurkha/Desktop/BaDou/Code/Course_CV/Week12/demo/mobilenet/mobilenet_1_0_224_tf.h5'
    model.load_weights(weights_path)

    # 加载并预处理图像
    img_path = 'C:/Users/Gurkha/Desktop/BaDou/Code/Course_CV/Week12/demo/mobilenet/elephant.jpg'  # 图像路径
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    print('输入图像的形状:', x.shape)

    # 进行预测
    preds = model.predict(x)
    print('预测结果:', decode_predictions(preds, top=5)[0])