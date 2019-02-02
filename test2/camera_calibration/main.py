
import os
import json
import math
import numpy as np


class_title_names = {'Outer': 'outer_perimeter',
                     'Inner': 'inner_perimeter',
                     'center points': 'center_points'}

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
    #break

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
    
for annotation in annotations:
    print("Filename", annotation['filename'])
    outer_corners = find_rectangle_corners(annotation['outer_perimeter'])
    inner_corners = find_rectangle_corners(annotation['inner_perimeter'])
    center_points = find_center_points(outer_corners, inner_corners, annotation['center_points'])





print("end")
quit()




"""
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
