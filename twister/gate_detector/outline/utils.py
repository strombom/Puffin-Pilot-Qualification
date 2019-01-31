
import os
import cv2
import json
import numpy as np


def load_data(data_path):

    def read_images(path):
        inputs, outputs = [], []

        for filename in os.listdir(path):
            if '_mask' in filename:
                continue

            image_filename = os.path.join(path, filename)
            outline_filename = image_filename.replace('.png', '_mask.png')

            image = cv2.imread(image_filename)
            outline = cv2.imread(outline_filename)

            image = image / 256.
            outline = outline / 256.
            outline = outline[:,:,0]
            outline = outline.reshape((outline.shape[0], outline.shape[1], 1))

            inputs.append(image)
            outputs.append(outline)
      
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        return inputs, outputs

    training_data = read_images(os.path.join('data', 'train'))
    validation_data = read_images(os.path.join('data', 'val'))

    return training_data, validation_data

def save_image(filename, image):
    cv2.imwrite(filename, image * 255)

def save_images(filename, images):
    y_count = len(images)
    x_count = len(images[0])

    height = images[0][0].shape[0]
    width = images[0][0].shape[1]

    image = np.zeros((height * y_count, width * x_count, 1))

    for row in range(y_count):
        for col in range(x_count):
            x, y = col * width, row * height
            image[y:y+height, x:x+width] = images[row][col]

    cv2.imwrite(filename, image * 255)





if __name__ == '__main__':

    training_data, validation_data = load_data('data')

    print(training_data[0].shape, training_data[1].shape)
    print(validation_data[0].shape, validation_data[1].shape)

