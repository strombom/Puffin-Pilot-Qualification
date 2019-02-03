
import os
import cv2
import json
import math
import pickle
import numpy as np


original_path = '/home/jst/development/data-puffin-pilot/Data_Training'
undistorted_path = '/home/jst/development/data-puffin-pilot/Undistorted'

camera_calibration = pickle.load(open("camera_calibration.pickle", "rb"))

rms = camera_calibration['rms']
camera_matrix = camera_calibration['camera_matrix']
dist_coefs = camera_calibration['dist_coefs']
rvecs = camera_calibration['rvecs']
tvecs = camera_calibration['tvecs']
image_size = (864, 1296)

map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, camera_matrix, image_size[::-1], cv2.CV_32FC1)

for filename in os.listdir(original_path):
    print(filename)
    img = cv2.imread(os.path.join(original_path, filename))
    dst = cv2.remap(img, map1, map2, cv2.INTER_CUBIC)
    cv2.imwrite(os.path.join(undistorted_path, filename), dst)
