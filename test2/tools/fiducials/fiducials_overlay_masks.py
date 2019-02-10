
import os
import sys
import cv2
import json
import math
import pickle
import random
import skimage
import numpy as np
from sklearn.cluster import KMeans
from seaborn import color_palette
from inspect import getsourcefile

sys.path.insert(0, "../../")
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

exclusions = [219,220,226,2411,2631,2713,2816,4067,5984,6328,6454,6543,6572,6600,6673,7643,8518,8757,8777,8797,8817,8825,8837,8846,8853,8861,8862,8865,8870,8876,8914,8942,9080,9243,9275,9282,9292,9294,9300,7313,9324]

annotations = []
for filename in os.listdir(metadata_path): #['IMG_1352.json']: # os.listdir(metadata_path):
    with open(os.path.join(metadata_path, filename)) as f:
        imnum = int(filename.replace('IMG_','').replace('.json',''))
        if imnum in exclusions:
            continue
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
    tops, bots = [], []
    lefts, rights = [], []

    widths = []
    heights = []

    #poss = []

    for annotation in annotations:
        fiducials = annotation['fiducials'].astype(np.int64)
        image_name = annotation['image_name']

        fiducials = fix_fiducials(fiducials, image_name)
        if fiducials is None:
            continue

        xs, ys = [], []
        for idx, fiducial in enumerate(fiducials):
            x, y = int(round(fiducial[0])), int(round(fiducial[1]))
            xs.append(x)
            ys.append(y)

        tops.append(min(ys))
        bots.append(max(ys))
        lefts.append(min(xs))
        rights.append(max(xs))

        widths.append(max(xs)-min(xs))
        heights.append(max(ys)-min(ys))


    print(widths)
    print(heights)

    print(min(widths), min(heights))
    print(max(widths), max(heights))
    quit()


    filename = os.path.join(original_path, image_name + '.JPG')
    input_image = cv2.imread(filename, 0)
    height, width = input_image.shape[0], input_image.shape[1]
    output_image = np.zeros((height, width, 3), dtype=np.uint8)

    colors = [(75,234,255),
              (255,75,213),
              (123,75,255),
              (75,255,117)]

    for x in lefts:
        cv2.line(output_image, (int(x), 0), (int(x), height-1), colors[0])
    for x in rights:
        cv2.line(output_image, (int(x), 0), (int(x), height-1), colors[1])
    for y in tops:
        cv2.line(output_image, (0, int(y)), (width-1, int(y)), colors[2])
    for y in bots:
        cv2.line(output_image, (0, int(y)), (width-1, int(y)), colors[3])

    #for pos in poss:
    #    x, y = pos
    #    cv2.circle(output_image, (int(x), int(y)), 1, (255), thickness = 1)

    cv2.imwrite('test.png', output_image)

    #print(poss)
    #print(poss.shape)

    """
    img_mask = np.zeros((height, width, 3), dtype=np.uint8)

    region = input_image[y-3:y+4,x-3:x+4]

    contrast = np.max(region) - np.min(region)

    mins.append(np.min(region))
    maxs.append(np.max(region))

    if contrast > 18:
        cv2.circle(img_mask, (int(x), int(y)), 1, (255, 255, 255), thickness = 2)
        img_mask[y-2:y+3,x-2:x+3] = (255, 255, 255)

        cv2.circle(output_image, (int(x), int(y)), 3, (255), thickness = 2)
        cv2.circle(output_image, (int(x), int(y)), 2, (0), thickness = 2)
    """

    """
    filename = os.path.join(original_path, image_name + '.JPG')
    input_image = cv2.imread(filename, 0)
    height, width = input_image.shape[0], input_image.shape[1]

    output_image = input_image.copy()
    img_mask = cv2.GaussianBlur(img_mask,(3,3),0)

    file_path = os.path.join(masks_path, image_name + '_mask.png')
    cv2.imwrite(file_path, img_mask)
    """

    #print(input_image[:].shape, img_mask[:,:,0].shape)
    #img_mask = (img_mask * 0.5).astype(np.uint8)
    #input_image[:] = cv2.subtract(input_image[:], img_mask[:,:,0])

    """
    cv2.imwrite(os.path.join('fiducial_maketest', image_name + '.jpg'), output_image)


    mins = np.array(mins)
    maxs = np.array(maxs)

    print(mins)
    print(maxs)
    print(maxs-mins)

    print(len(mins))

    print(sorted(mass))

    #hist = np.histogram(maxs-mins, bins=20)

    import matplotlib.pyplot as plt
    #a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))
    plt.hist(maxs-mins, bins=100)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
    """

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
