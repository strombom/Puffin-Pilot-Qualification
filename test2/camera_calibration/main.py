
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
        if abs(a) > math.pi / 4:
            corners.append(points[idx])
        direction = new_direction
    if len(corners) != 4:
        print("Error, too many corners")
        quit()
    corners[3][0] = points[-1][0]
    return corners

def find_center_points(outer_corners, inner_corners, points):
    print(outer_corners)
    print(inner_corners)
    print(points)
    quit()

for annotation in annotations:

    outer_corners = find_rectangle_corners(annotation['outer_perimeter'])
    inner_corners = find_rectangle_corners(annotation['inner_perimeter'])
    center_points = find_center_points(outer_corners, inner_corners, annotation['center_points'])

    print("end")
    quit()

