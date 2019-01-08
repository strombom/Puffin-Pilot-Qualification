
import os
import cv2
import json
import numpy as np


def load_data(data_path):

    def read_images(path):
        inputs, outputs = [], []

        annotations = json.load(open(os.path.join(data_path, path, 'via_region_data.json')))
        annotations = annotations['_via_img_metadata']
        annotations = list(annotations.values())

        for annotation in annotations:
            image_filename = annotation['filename']
            outline_filename = image_filename.replace('.png', '_mask.png')

            image = cv2.imread(os.path.join(data_path, path, image_filename))
            outline = cv2.imread(os.path.join(data_path, path, outline_filename))

            image = image / 256
            outline = outline / 256
            outline = outline[:,:,0]
            outline = outline.reshape((outline.shape[0], outline.shape[1], 1))

            inputs.append(image)
            outputs.append(outline)

        inputs = np.array(inputs)
        outputs = np.array(outputs)

        return inputs, outputs

    training_data = read_images('train')
    validation_data = read_images('val')

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

