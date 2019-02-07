
import os
import cv2
import json
import math
import pickle
import random
import skimage
import numpy as np


metadata_path = 'ds/ann'

annotations = []
for filename in os.listdir(metadata_path):
    with open(os.path.join(metadata_path, filename)) as f:
        file_data = json.load(f)
        annotation = {'fiducials': []}
        for obj in file_data['objects']:
            class_title = obj['classTitle']
            points = obj['points']['exterior']
            if class_title == 'fiducial':
                annotation['fiducials'].extend(points)
        for key in annotation:
            annotation[key] = np.array(annotation[key])
        annotation['image_name'] = filename.replace('.json', '')
        annotations.append(annotation)


def magn(v):
    return (v[0] * v[0] + v[1] * v[1]) ** 0.5

min_dists = []

for annotation in annotations:
    fiducials = annotation['fiducials']
    image_name = annotation['image_name']

    min_distance = 1e6
    for idx_i in range(fiducials.shape[0]):
        for idx_j in range(idx_i + 1, fiducials.shape[0]):

            l = magn(fiducials[idx_i] - fiducials[idx_j])
            if l < min_distance:
                min_distance = l

    if min_dists and min_distance < min(min_dists):
        print("Liten", image_name, min_distance)
    min_dists.append(min_distance)

print("Min/max distance", min(min_dists), max(min_dists))


