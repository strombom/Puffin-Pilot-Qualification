
import os
import cv2
import json
import math
import pickle
import random
import skimage
import numpy as np
from sklearn.cluster import KMeans
from seaborn import color_palette

from gate_model import GateModel


undistorted_path = '../../../../data-puffin-pilot/Undistorted'
original_path = '../../../../data-puffin-pilot/Data_Training'
masks_path = '../../../../data-puffin-pilot/Fiducials_Masks'

metadata_path = 'ds/ann'

camera_calibration = pickle.load(open("../../camera_calibration/camera_calibration.pickle", "rb"))

camera_matrix = camera_calibration['camera_matrix']
dist_coefs = camera_calibration['dist_coefs']
image_size = camera_calibration['image_size']

gate_model = GateModel(camera_matrix, dist_coefs)

map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, camera_matrix, image_size, cv2.CV_32FC1)


annotations = []
for filename in os.listdir(metadata_path): #['IMG_1352.json']: # os.listdir(metadata_path):
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

def fix_fiducials(fiducials, image_name):
    if len(fiducials) < 4:
        return None
    elif len(fiducials) > 4:
        return fiducials
    return gate_model.fiducials_from_corners(fiducials)

def make_masks():
    for annotation in annotations:
        fiducials = annotation['fiducials'].astype(np.int64)
        image_name = annotation['image_name']

        print(image_name)
        fiducials = fix_fiducials(fiducials, image_name)
        if fiducials is None:
            continue

        filename = os.path.join(original_path, image_name + '.JPG')
        im = skimage.io.imread(filename)
        height, width = im.shape[0], im.shape[1]
        img_mask = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(2):
            for idx, fiducial in enumerate(fiducials):
                x, y = int(round(fiducial[0])), int(round(fiducial[1]))
                #cv2.circle(img_mask, (int(x), int(y)), 1, (255, 255, 255), thickness = 2)
                img_mask[y-2:y+3,x-2:x+3] = (255, 255, 255)
            img_mask = cv2.GaussianBlur(img_mask,(3,3),0)

        file_path = os.path.join(masks_path, image_name + '_mask.png')
        cv2.imwrite(file_path, img_mask)


make_masks()



"""
filename = image_name + ".JPG"
im2 = cv2.imread(os.path.join(original_path, filename))
for fiducial in estimated_fiducials:
    x, y = int(fiducial[0]), int(fiducial[1])
    cv2.circle(im2, (x, y), 4, (0, 0, 255))
cv2.imwrite(filename.replace('.', '_fid.'), im2)
quit()

filename = image_name + ".JPG"
im2 = cv2.imread(os.path.join(original_path, filename), 0)
dst = cv2.remap(im2, map_x, map_y, cv2.INTER_CUBIC)
cv2.imwrite(filename.replace('.', '_.'), dst)

print('[817][118]'
distorted = np.empty((1, 1, 2))
distorted[0][0] = (118 ,817)
object_points = np.ones((distorted.shape[0], 1, 3))
object_points[:,:,0:2] = cv2.undistortPoints(distorted, camera_matrix, dist_coefs)
undistorted_image_points, jacobian = cv2.projectPoints(object_points, (0,0,0), (0,0,0), camera_matrix, None)
distorted_image_points, jacobian = cv2.projectPoints(object_points, (0,0,0), (0,0,0), camera_matrix, dist_coefs)
"""
