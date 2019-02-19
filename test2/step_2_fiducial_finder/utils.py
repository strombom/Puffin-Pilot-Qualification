
import os
import cv2
import json
import numpy as np


def save_image(filename, image):
    cv2.imwrite(filename, image * 255)

def save_images(filename, images):
    y_count = len(images)
    x_count = len(images[0])

    height = images[0][0].shape[0]
    width = images[0][0].shape[1]

    image = np.zeros((height * y_count, width * x_count, 1), dtype = np.uint8)

    for row in range(y_count):
        for col in range(x_count):
            x, y = col * width, row * height
            image[y:y+height, x:x+width] = images[row][col] * 255

    cv2.imwrite(filename, image)


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    number_size = 4.0
    if K.floatx() == 'float16':
         number_size = 2.0
    if K.floatx() == 'float64':
         number_size = 8.0

    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes



if __name__ == '__main__':

    training_data, validation_data = load_data('data')

    print(training_data[0].shape, training_data[1].shape)
    print(validation_data[0].shape, validation_data[1].shape)

