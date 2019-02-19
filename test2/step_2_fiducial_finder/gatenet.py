
from . import utils

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Dropout, ReLU
from keras.layers import UpSampling2D, Reshape, MaxPooling2D
from keras.layers import add, concatenate
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model


def gatenet(debug_print = False):
    input_shape = (864, 1296, 1)
    n_filters = [8, 4, 2, 2, 2]

    def _conv_block(inputs, filters, kernel, strides):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(6.0)(x)
        return x

    def _bottleneck(inputs, filters, kernel, t, s, r=False):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        tchannel = K.int_shape(inputs)[channel_axis] * t
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(6.0)(x)
        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        if r:
            x = add([x, inputs])
        return x

    def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
        x = _bottleneck(inputs, filters, kernel, t, strides)
        for i in range(1, n):
            x = _bottleneck(x, filters, kernel, t, 1, True)
        return x

    def conv2d_block(input_tensor, n_filters, batchnorm = True):
        x = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(input_shape)
    conv1 = conv2d_block(inputs, n_filters[0])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.02)(pool1)
    
    conv2 = conv2d_block(drop1, n_filters[1])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.05)(pool2)
    
    conv3 = conv2d_block(pool2, n_filters[2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.05)(pool3)
    
    conv5 = conv2d_block(drop3, n_filters[4])
    drop5 = Dropout(0.05)(conv5)

    up7 = Conv2DTranspose(n_filters[2], (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = Dropout(0.05)(merge7)
    conv7 = conv2d_block(merge7, n_filters[3])

    up8 = Conv2DTranspose(n_filters[1], (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = Dropout(0.05)(merge8)
    conv8 = conv2d_block(merge8, n_filters[2])

    up9 = Conv2DTranspose(n_filters[0], (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    merge9 = Dropout(0.02)(merge9)
    conv9 = conv2d_block(merge9, n_filters[1])

    conv9 = Conv2D(n_filters[2], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    def dice_loss():
        def dice(y_true, y_pred, smooth=1e-6):
            y_true_f = K.batch_flatten(y_true)
            y_pred_f = K.batch_flatten(y_pred)
            intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
            union = K.sum(y_true_f, axis=1, keepdims=True) + \
                    K.sum(y_pred_f, axis=1, keepdims=True) + \
                    smooth
            return 2-K.mean(intersection / union)
        return dice
    
    model_loss= dice_loss()

    model.compile(optimizer = Adam(lr=1e-5), loss = model_loss, metrics = ['accuracy'])
    
    if debug_print:
        print(model.summary())
        print("Num params", model.count_params())
        print("Mem req", utils.get_model_memory_usage(batch_size = 8, model = model))

    return model

