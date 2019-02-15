
import os
import cv2
import time
import random
import resource
import numpy as np

#import keras
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *

import utils
from gatenet import gatenet





original_path = '../../../data-puffin-pilot/Data_Training'
masks_path = '../../../data-puffin-pilot/Fiducials_Masks'


def read_images(image_names):
    inputs, outputs = [], []

    for image_name in image_names:
        input_filepath = os.path.join(original_path, image_name + '.JPG')
        output_filepath = os.path.join(masks_path, image_name + '_mask.png')

        input_image = cv2.imread(input_filepath)
        output_image = cv2.imread(output_filepath)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)

        #image = image / 256
        #outline = outline / 256
        #output_image = output_image[:,:,0]
        input_image = input_image.reshape((input_image.shape[0], input_image.shape[1], 1))
        output_image = output_image.reshape((output_image.shape[0], output_image.shape[1], 1))

        #input_image = input_image.astype(np.float16)
        #output_image = output_image.astype(np.float16)
        #input_image /= 255.0
        #output_image /= 255.0

        inputs.append(input_image)
        outputs.append(output_image)

    inputs = np.array(inputs)
    outputs = np.array(outputs)

    return inputs, outputs


def load_data():

    image_names = []
    for filename in os.listdir(masks_path):
        image_names.append(filename.replace('_mask.png', ''))

    random.seed(2)

    training_names, validation_names = [], []
    for image_name in image_names:
        if random.random() < 0.95:
            training_names.append(image_name)
        else:
            validation_names.append(image_name)

    training_data = read_images(training_names)
    validation_data = read_images(validation_names)

    return training_data, validation_data


training_data, validation_data = load_data()

model = gatenet(debug_print = True)




class ProgressImageSaver(callbacks.Callback):
    def on_train_begin(self, logs={}):
        #print("ProgressImageSaver begin")
        self.imnm = 0

    def on_epoch_end(self, batch, logs={}):
        predictions = model.predict(validation_data[0])
        images_validation = predictions[0:6]
        predictions = model.predict(training_data[0])
        images_training = predictions[0:6]

        filename = 'img_%d.png' % (self.imnm,)
        self.imnm += 1

        utils.save_images(filename, [images_validation, images_training])

class MemoryCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        #print("mem", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        pass

memory_callback = MemoryCallback()
progress_image_saver = ProgressImageSaver()
tensorboard = callbacks.TensorBoard(log_dir='logdir', histogram_freq=0)
checkpoint_saver = callbacks.ModelCheckpoint('logdir/checkpoint', 
                                             save_weights_only=False,
                                             verbose=0)




try:
    model.load_weights('logdir/checkpoint')
except:
    pass

#for x in model.layers:
#    x.trainable = False

def timeit():
    import time
    indat = np.array([validation_data[0][0]])
    for i in range(10):
        start_time = time.time()
        predictions = model.predict(indat)
        print("time", time.time() - start_time)
    print(predictions.shape)
    quit()
#timeit()

batch_size = 8


model.fit(x = training_data[0], 
          y = training_data[1], 
          epochs = 10000,
          batch_size = batch_size,
          #steps_per_epoch = 100,
          #validation_steps = 1,
          validation_data = validation_data,
          callbacks=[tensorboard, progress_image_saver, checkpoint_saver])

quit()



data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=5,
                     width_shift_range=.1,
                     height_shift_range=.1,
                     zoom_range=0.05,
                     horizontal_flip=True
                     )

train_image_datagen = ImageDataGenerator(**data_gen_args)
train_mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1 # Provide the same seed and keyword arguments to the fit and flow methods
train_image_datagen.fit(training_data[0], augment=True, seed=seed)
train_mask_datagen.fit(training_data[1], augment=True, seed=seed)

train_image_generator = train_image_datagen.flow(training_data[0], seed=seed, batch_size=batch_size)
train_mask_generator = train_mask_datagen.flow(training_data[1], seed=seed, batch_size=batch_size)

train_generator = zip(train_image_generator, train_mask_generator)


# fits the model on batches with real-time data augmentation:
model.fit_generator(train_generator,
                    epochs = 10000,
                    #batch_size = 100,
                    steps_per_epoch = 4 * len(training_data[0]) / batch_size,
                    validation_steps = 1,
                    validation_data = validation_data, #val_generator,
                    callbacks=[memory_callback, progress_image_saver, checkpoint_saver],
                    workers=1,
                    use_multiprocessing=False) #tensorboard, 


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


