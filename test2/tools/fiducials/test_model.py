
import os
import cv2
import time
import numpy as np
from itertools import izip_longest

from gatenet import gatenet


original_path = '../../../../data-puffin-pilot/Data_Training'


model = gatenet()


try:
    model.load_weights('logdir/checkpoint_20190206_loss449')
except:
    print("Failed to load model")
    quit()


image_names = []
for filename in os.listdir(original_path):
    image_names.append(filename.replace('.JPG', ''))


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)

batch_size = 5
for batch_idx, image_name_batch in enumerate(grouper(image_names, batch_size)):
    print("Saving img", batch_idx * batch_size)

    images = []
    for image_name in image_name_batch:
        image_filepath = os.path.join(original_path, image_name + '.JPG')
        image = cv2.imread(image_filepath)
        images.append(image)
    images = np.array(images)

    predictions = model.predict(images)

    for idx in range(batch_size):
        image = images[idx]

        prediction = (predictions[idx] * 128).astype(np.uint8)
        image[:,:,0] = cv2.subtract(image[:,:,0], prediction[:,:,0])
        image[:,:,1] = cv2.subtract(image[:,:,1], prediction[:,:,0])
        image[:,:,2] = cv2.add(image[:,:,2], prediction[:,:,0])

        filename = image_name_batch[idx] + '_point.jpg'
        cv2.imwrite(os.path.join('fiducial_img', filename), image)

        prediction = (predictions[idx] * 255).astype(np.uint8)
        filename = image_name_batch[idx] + '_mask.png'
        cv2.imwrite(os.path.join('fiducial_mask', filename), prediction)