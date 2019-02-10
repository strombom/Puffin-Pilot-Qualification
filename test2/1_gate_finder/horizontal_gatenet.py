



"""
import keras
from keras.utils.vis_utils import plot_model

from keras.optimizers import Adam

from keras.models import *
from keras.layers import *
from keras.optimizers import *


from batchnormfp16 import BatchNormalizationF16
"""

import utils




#import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2DTranspose
from keras.layers import Activation, BatchNormalization, Dropout, ReLU
from keras.layers import UpSampling2D, Reshape, MaxPooling2D
from keras.layers import add, concatenate
from keras.optimizers import Adam, SGD
from keras.utils import multi_gpu_model


def gatenet():
    input_shape = (864, 1296, 1)
    #n_filters = [8, 8, 2, 2, 8]
    n_filters = [8, 4, 2, 2, 2]

    #K.set_floatx('float32')
    #K.set_epsilon(1e-4)
    
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
    
    #conv4 = Conv2D(n_filters[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv4 = Conv2D(n_filters[3], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv2d_block(drop3, n_filters[4])
    drop5 = Dropout(0.05)(conv5)

    #up6 = Conv2D(n_filters[4], 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = concatenate([drop4,up6], axis = 3)
    #conv6 = Conv2D(n_filters[4], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(n_filters[4], 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

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
    #model = multi_gpu_model(model, gpus=2)

    def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
        ''' 
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the `channels_last` format.
      
        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
            epsilon: Used for numerical stability to avoid divide by zero errors
        
        # References
            V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
            https://arxiv.org/abs/1606.04797
            More details on Dice loss formulation 
            https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
            
            Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
        '''
        
        # skip the batch and class axis for calculating Dice score
        axes = tuple(range(1, len(y_pred.shape)-1)) 
        numerator = 2. * np.sum(y_pred * y_true, axes)
        denominator = np.sum(np.square(y_pred) + np.square(y_true), axes)
        
        return 1 - np.mean(numerator / (denominator + epsilon)) # average over classes and batch

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
    
    #def focal_loss(gamma=2, alpha=0.75):
    #    def focal_loss_int(y_true, y_pred):
    #        eps = 1e-12
    #        y_pred=K.clip(y_pred,eps,1.-eps)
    #        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    #        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    #        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    #    return focal_loss_int

    #def focal_loss(gamma=2., alpha=.25):
    #    def focal_loss_fixed(y_true, y_pred):
    #        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    #        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))#

    #        pt_1 = K.clip(pt_1, 1e-3, .999)
    #        pt_0 = K.clip(pt_0, 1e-3, .999)

    #        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    #    return focal_loss_fixed


    model_loss= dice_loss()

    #Adam(1e-4, epsilon = 1e-4)

    model.compile(optimizer = Adam(lr=0.000001), loss = model_loss, metrics = ['accuracy'])
    #model.summary()

    #quit()

    """
    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, 32, (3, 3), strides=(1, 1))
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=1)
    x = UpSampling2D((2, 2))(x)
    x = _conv_block(x, 12, (1, 1), strides=(1, 1))
    #x = Conv2D(3, (1, 1), strides=(1, 1), padding='same')(UpSampling2D((2, 2))(x))
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(3, (1, 1), strides=(1, 1), padding='same')(x)
    x = Activation('softmax', name='softmax')(x)

    output = Reshape((288, 360, 3))(x)

    model = Model(inputs, output)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    """

    #plot_model(model, to_file='gatenet.png', show_shapes=True)

    print(model.summary())
    print("Num params", model.count_params())
    print("Mem req", utils.get_model_memory_usage(batch_size = 8, model = model))
    #quit()

    #writer = tf.summary.FileWriter(logdir='logdir', graph=K.get_session().graph)
    #writer.flush()

    return model

