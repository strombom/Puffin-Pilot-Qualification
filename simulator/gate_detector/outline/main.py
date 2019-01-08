
import time

import keras
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dropout, ReLU, UpSampling2D
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

from keras.models import *
from keras.layers import *
from keras.optimizers import *

import utils


def gatenet():
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

    input_shape = (288, 360, 3)

    n_filters = 16

    def conv2d_block(input_tensor, n_filters, batchnorm = True):
        x = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(input_tensor)
        x = BatchNormalization()(x)
        x = Conv2D(n_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x)
        return x

    inputs = Input(input_shape)
    conv1 = conv2d_block(inputs, n_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    conv2 = conv2d_block(pool1, n_filters*2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.25)(pool2)
    conv3 = conv2d_block(pool2, n_filters*4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.25)(pool3)
    
    #conv4 = Conv2D(n_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    #conv4 = Conv2D(n_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = Dropout(0.5)(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = conv2d_block(drop3, n_filters*8)
    drop5 = Dropout(0.25)(conv5)

    #up6 = Conv2D(n_filters*8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    #merge6 = concatenate([drop4,up6], axis = 3)
    #conv6 = Conv2D(n_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    #conv6 = Conv2D(n_filters*8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    merge7 = concatenate([conv3,up7], axis = 3)
    merge7 = Dropout(0.25)(merge7)
    conv7 = conv2d_block(merge7, n_filters*4)

    up8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    merge8 = Dropout(0.25)(merge8)
    conv8 = conv2d_block(merge8, n_filters*2)

    up9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    merge9 = Dropout(0.25)(merge9)
    conv9 = conv2d_block(merge9, n_filters*1)

    conv9 = Conv2D(n_filters//2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(), loss = 'binary_crossentropy', metrics = ['accuracy'])
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

    plot_model(model, to_file='gatenet.png', show_shapes=True)

    import tensorflow as tf
    writer = tf.summary.FileWriter(logdir='logdir', graph=K.get_session().graph)
    writer.flush()

    return model

def load_data():
    data_path = 'data'
    training_data, validation_data = utils.load_data(data_path)
    return training_data, validation_data



model = gatenet()

training_data, validation_data = load_data()



class ProgressImageSaver(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        #print("ProgressImageSaver begin")
        self.imnm = 0
        self.time = int(time.time())

    def on_batch_end(self, batch, logs={}):
        if int(time.time()) > self.time + 5:
            predictions = model.predict(validation_data[0])
            images_validation = predictions[0:6]
            predictions = model.predict(training_data[0])
            images_training = predictions[0:6]

            filename = 'img_%d.png' % (self.imnm,)
            self.imnm += 1

            utils.save_images(filename, [images_validation, images_training])
            self.time = int(time.time())


cp_callback = tf.keras.callbacks.ModelCheckpoint('logdir/checkpoint', 
                                                 save_weights_only=True,
                                                 verbose=0)
tensorboard = TensorBoard(log_dir='logdir')
progress_image_saver = ProgressImageSaver()

try:
    model.load_weights('logdir/checkpoint')
except:
    pass



data_gen_args = dict(#featurewise_center=True,
                     #featurewise_std_normalization=True,
                     rotation_range=30,
                     width_shift_range=.15,
                     height_shift_range=.15,
                     #width_shift_range=0.1,
                     #height_shift_range=0.1,
                     #zoom_range=0.2,
                     horizontal_flip=True
                     )

train_image_datagen = ImageDataGenerator(**data_gen_args)
train_mask_datagen = ImageDataGenerator(**data_gen_args)
#val_image_datagen = ImageDataGenerator(**data_gen_args)
#val_mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1 # Provide the same seed and keyword arguments to the fit and flow methods
train_image_datagen.fit(training_data[0], augment=True, seed=seed)
train_mask_datagen.fit(training_data[1], augment=True, seed=seed)
#val_image_datagen.fit(validation_data[0], augment=True, seed=seed)
#val_mask_datagen.fit(validation_data[1], augment=True, seed=seed)

train_image_generator = train_image_datagen.flow(training_data[0], seed=seed, batch_size=500)
train_mask_generator = train_mask_datagen.flow(training_data[1], seed=seed, batch_size=500)
#val_image_generator = val_image_datagen.flow(validation_data[0], seed=seed)
#val_mask_generator = val_mask_datagen.flow(validation_data[1], seed=seed)

train_generator = zip(train_image_generator, train_mask_generator)
#val_generator = zip(val_image_generator, val_mask_generator)

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    #x = training_data[0], 
                    #y = training_data[1], 
                    epochs = 10000,
                    #batch_size = 100,
                    steps_per_epoch = 20, #len(x_train) / 32,
                    validation_steps = 1,
                    validation_data = validation_data, #val_generator,
                    callbacks=[tensorboard, progress_image_saver, cp_callback])


#datagen.flow(x_train, y_train, batch_size=32),
#          steps_per_epoch=len(x_train) / 32, epochs=epochs)
"""
model.fit(x = training_data[0], 
          y = training_data[1], 
          epochs = 500,
          #batch_size = 100,
          steps_per_epoch = 60,
          validation_steps = 1,
          validation_data = validation_data,
          callbacks=[tensorboard, progress_image_saver, cp_callback])
"""

