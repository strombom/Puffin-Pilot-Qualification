
import os
import cv2
import json
import math
import pickle
import numpy as np


original_path = '../../data-puffin-pilot/Data_Training'
undistorted_path = '../../data-puffin-pilot/Undistorted'

camera_calibration = pickle.load(open("camera_calibration.pickle", "rb"))

camera_matrix = camera_calibration['camera_matrix']
dist_coefs = camera_calibration['dist_coefs']
image_size = camera_calibration['image_size']

map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, camera_matrix, image_size, cv2.CV_32FC1)

for filename in os.listdir(original_path):
    print(filename)
    img = cv2.imread(os.path.join(original_path, filename))
    dst = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(undistorted_path, filename.replace('.JPG', 'u.jpg')), dst)
