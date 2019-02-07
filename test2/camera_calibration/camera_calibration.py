
import os
import cv2
import json
import math
import pickle
import numpy as np

class_title_names = {'Outer': 'outer_perimeter',
                     'Inner': 'inner_perimeter',
                     'center points': 'center_points'}
calibration_path = '../../data-puffin-pilot/Data_Training'


annotations = []
annotation_path = os.path.join('calibration_data', 'ds', 'ann')
for filename in os.listdir(annotation_path):
    with open(os.path.join(annotation_path, filename)) as f:
        file_data = json.load(f)
        annotation = {'center_points': []}
        for obj in file_data['objects']:
            class_title = class_title_names[obj['classTitle']]
            points = obj['points']['exterior']
            if class_title == 'center_points':
                annotation['center_points'].extend(points)
            else:
                annotation[class_title] = points
        for key in annotation:
            annotation[key] = np.array(annotation[key])
        annotation['filename'] = filename
        annotations.append(annotation)


def vector_intersection_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)

def find_rectangle_corners(points):
    corners = [points[0]]
    direction = points[1] - points[0]
    for idx in range(2, len(points)):
        new_direction = points[idx] - points[idx - 1]
        a = vector_intersection_angle(new_direction, direction)
        if abs(a) > math.pi / 6:
            corners.append(points[idx-1])
        direction = new_direction
    if len(corners) != 4:
        print("Error, too many corners")
        quit()
    corners[0][0] = points[-1][0]
    return corners

def find_center_points(outer_corners, inner_corners, center_points):
    target_points = np.zeros((8, 2))
    for idx in range(4):
        target_points[idx * 2 + 0] = (outer_corners[idx] + outer_corners[(idx + 1) % 4]) / 2
        target_points[idx * 2 + 1] = (inner_corners[idx] + inner_corners[(idx + 1) % 4]) / 2
    ordered_points = np.zeros((8, 2))
    for center_point in center_points:
        distances = (target_points - center_point) ** 2
        distances = np.sqrt(distances[:,0] + distances[:,1])
        closest_idx = np.argmin(distances)
        ordered_points[closest_idx] = center_point
    return ordered_points

def create_calibration_points(outer_corners, inner_corners, center_points, goal_size, goal_post_width):
    bottom_right_corner_z = -0.020 # Special case for lower right corner which seems to stick out a bit

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # Corners
    for idx in range(4):
        # Corner position
        row = (idx // 2) * 2 - 1
        col = ((idx + idx // 2) % 2) * 2 - 1

        # Add outer corner
        outer_x = col * goal_size / 2
        outer_y = row * goal_size / 2
        outer_z = 0.0
        if idx == 2:
            outer_z = bottom_right_corner_z
        objpoints.append((outer_x, outer_y, outer_z))
        imgpoints.append(outer_corners[idx])

        # Add inner corner
        inner_x = outer_x - col * goal_post_width
        inner_y = outer_y - row * goal_post_width
        objpoints.append((inner_x, inner_y, 0))
        imgpoints.append(inner_corners[idx])

        # Center point position mask
        row, col = 1 - idx % 2, idx % 2

        # Add outer center point if exists
        if np.all(center_points[idx * 2]):
            center_x, center_y = outer_x * col, outer_y * row
            center_z = 0.0
            objpoints.append((center_x, center_y, center_z))
            imgpoints.append(center_points[idx * 2])

        # Add inner center point if exists
        if np.all(center_points[idx * 2 + 1]):
            center_x, center_y = inner_x * col, inner_y * row
            center_z = 0.0
            objpoints.append((center_x, center_y, center_z))
            imgpoints.append(center_points[idx * 2 + 1])

    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints = np.array(imgpoints, dtype=np.float32)
    return objpoints, imgpoints

def get_image_size(filename):
    img = cv2.imread(filename, 0)
    return img.shape


calibration_points = {'obj': [], 'img': []}
for annotation in annotations:
    #print("Filename", annotation['filename'])
    outer_corners = find_rectangle_corners(annotation['outer_perimeter'])
    inner_corners = find_rectangle_corners(annotation['inner_perimeter'])
    center_points = find_center_points(outer_corners, inner_corners, annotation['center_points'])

    goal_size = 11 * 0.3048 # Assumed, 11 ft, 3.344 m
    goal_post_width = 0.4525 # Optimized

    objpoints, imgpoints = create_calibration_points(outer_corners, inner_corners, center_points, goal_size, goal_post_width)
    calibration_points['obj'].append(objpoints)
    calibration_points['img'].append(imgpoints)

camera_matrix = np.array([[1.11615226e+03, 0.00000000e+00, 6.42354107e+02],
                          [0.00000000e+00, 1.11453403e+03, 4.53277504e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coefs = np.array([[-0.15204789,  0.14581479, -0.00107285, -0.00019929,  0.04981551]])

image_size = (1296, 864)
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(calibration_points['obj'], calibration_points['img'], image_size, camera_matrix, dist_coefs, None, None, flags = cv2.CALIB_USE_INTRINSIC_GUESS)
print("rms", goal_post_width, rms)

camera_calibration = {'rms': rms,
                      'camera_matrix': camera_matrix,
                      'dist_coefs': dist_coefs,
                      'rvecs': rvecs,
                      'tvecs': tvecs,
                      'image_size': image_size}

pickle.dump(camera_calibration, open( "camera_calibration.pickle", "wb" ) )



"""

import time
print("undistort")
loops = 100
totaltime = 0
for i in range(loops):
    ts = time.time()
    dst = cv2.remap(img, map1, map2, cv2.INTER_CUBIC)
    totaltime += time.time() - ts
print(totaltime / loops)
quit()


import random
import skimage
from seaborn import color_palette
palette = color_palette("husl", 8)
def get_color(idx):
    return (palette[idx][0] * 255, palette[idx][1] * 255, palette[idx][2] * 255)
im = skimage.io.imread("/home/jst/development/data-puffin-pilot/CameraCalibration/" + annotation['filename'].replace('.json', '.JPG'))

for idx, center_point in enumerate(center_points):
    y, x = int(center_point[1]), int(center_point[0])
    rr, cc = skimage.draw.circle(y, x, 4)
    im[rr, cc] = get_color(idx)

for idx in range(4):
    y, x = int(outer_corners[idx][1]), int(outer_corners[idx][0])
    print(x, y)
    rr, cc = skimage.draw.circle(y, x, 4)
    im[rr, cc] = get_color(idx)

for idx in range(4):
    y, x = int(inner_corners[idx][1]), int(inner_corners[idx][0])
    print(x, y)
    rr, cc = skimage.draw.circle(y, x, 4)
    im[rr, cc] = get_color(idx + 4)
skimage.io.imsave(annotation['filename'].replace('.json', '.JPG'), (im).astype(np.uint8))
"""
