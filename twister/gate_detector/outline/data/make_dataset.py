
import os
import cv2
import json
import random
import numpy as np
from numba import jit, njit


val_path = 'val'
train_path = 'train'
supervisely_path = 'supervisely'

annotation_path = os.path.join('supervisely', 'ds', 'ann')
camera_image_path = os.path.join('supervisely', 'ds', 'img')


for path in (val_path, train_path):
    try:
        os.makedirs(path)
    except Exception:
        pass


for filename in os.listdir(annotation_path):
    annotation = json.load(open(os.path.join(annotation_path, filename)))
    if len(annotation['objects']) < 1:
        continue

    height, width = 480, 640
    img_mask = np.zeros((height, width, 3))
    img_mask_canvas = np.zeros((240, 320, 3))
    img_camera_canvas = np.zeros((240, 320, 3))

    for obj in annotation['objects']:
        points = obj['points']['exterior']

        line_count = len(points)
        lines = np.zeros((line_count, 2, 2), dtype=np.int)
        lines[-1] = ((points[0][0], points[0][1]), (points[-1][0], points[-1][1]))
        for line_idx in range(line_count - 1):
            lines[line_idx] = ((points[line_idx][0], points[line_idx][1]), (points[line_idx+1][0], points[line_idx+1][1]))

        for line_idx in range(line_count):
            line = lines[line_idx]
            p1 = (line[0][0], line[0][1])
            p2 = (line[1][0], line[1][1])

            if (p1[0] < 5 and p2[0] < 5) or \
                (p1[1] < 5 and p2[1] < 5) or \
                (p1[0] > width - 5 and p2[0] > width - 5) or \
                (p1[1] > height - 5 and p2[1] > height - 5):
                print(p1, p2, filename)

                continue

            #print( (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 0))
            cv2.line(img_mask, p1, p2, (255, 255, 255), thickness=2)

    if random.random() < 0.8:
        path = train_path
    else:
        path = val_path

    img_mask = cv2.GaussianBlur(img_mask,(3,3),0)
    img_mask = cv2.resize(img_mask, (320, 240), interpolation = cv2.INTER_AREA)
    #img_mask_canvas[24:288-24,:] = img_mask
    #img_mask = img_mask_canvas
    mask_filename = filename.replace('.json', '_mask.png')
    cv2.imwrite(os.path.join(path, mask_filename), img_mask)

    camera_filename = filename.replace('.json', '.jpg')
    camera_path = os.path.join(camera_image_path, camera_filename)
    img_camera = cv2.imread(camera_path)
    img_camera = cv2.resize(img_camera, (320, 240), interpolation = cv2.INTER_AREA)
    #img_camera_canvas[24:288-24,:] = img_camera
    #img_camera = img_camera_canvas
    camera_filename = filename.replace('.json', '.png')

    cv2.imwrite(os.path.join(path, camera_filename), img_camera)
