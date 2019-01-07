
import os
import cv2
import json
import numpy as np
from numba import jit, njit

for data_path in ['val', 'train']:

    annotations = json.load(open(os.path.join(data_path, "via_region_data.json")))
    annotations = annotations['_via_img_metadata']
    annotations = list(annotations.values())

    for annotation in annotations:
        image_filename = annotation['filename']

        #def transform_points(points_x, points_y):
        #    scale = 1 #360 / 1024
        #    offset_y = (1024 - 819.2) / 2 #(360 - 288) / 2
        #    for idx in range(len(points_x)):
        #        points_x[idx] = points_x[idx] * scale
        #        points_y[idx] = points_y[idx] * scale - offset_y
        #    return points_x, points_y

        #image = cv2.imread(os.path.join(data_path, image_filename))
        #height, width = image.shape[:2]
        #cv2.imshow("im", image)
        #cv2.waitKey()
        height, width = 1024, 1024

        outline = np.zeros((height, width, 3))
        
        polygons = [region['shape_attributes'] for region in annotation['regions']]
        for polygon in polygons:
            #points_x, points_y = transform_points(polygon['all_points_x'], polygon['all_points_y'])
            points_x, points_y = polygon['all_points_x'], polygon['all_points_y']
            line_count = len(points_x)
            lines = np.zeros((line_count, 2, 2), dtype=np.int)
            lines[-1] = [[points_x[0], points_y[0]], [points_x[-1], points_y[-1]]]
            for line_idx in range(line_count - 1):
                lines[line_idx] = [[points_x[line_idx], points_y[line_idx]], [points_x[line_idx+1], points_y[line_idx+1]]]
            
            for line_idx in range(line_count):
                line = lines[line_idx]
                p1 = (line[0][0], line[0][1])
                p2 = (line[1][0], line[1][1])
                #print( (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 0))
                cv2.line(outline, p1, p2, (255, 255, 255), thickness=5)

        #outline = cv2.GaussianBlur(outline,(5,5),0)
        outline = cv2.GaussianBlur(outline,(3,3),0)
        
        outline = cv2.resize(outline, (360, 360), interpolation = cv2.INTER_AREA)
        outline = outline[36:360-36,:]

        mask_filename = image_filename.replace('.png', '_mask.png')

        cv2.imwrite(os.path.join(data_path, mask_filename), outline)
        #outline = outline / 255.0
        #cv2.imshow("im", outline)
        #cv2.waitKey()
        #quit()





quit()










@jit(nopython=True)
def point_to_line_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the 
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    U, V = line[1] - line[0], line[0] - point
    cross = U[0] * V[1] - U[1] * V[0]
    segment_dist = abs(cross) / np.linalg.norm(unit_line)

    diff = (norm_unit_line[0] * (point[0] - line[0][0])) + (norm_unit_line[1] * (point[1] - line[0][1]))

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(np.linalg.norm(line[0] - point), np.linalg.norm(line[1] - point))

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist


@jit(nopython=True)
def draw_outline(outline, line):
    for y in range(outline.shape[0]):
        for x in range(outline.shape[1]):
            distance = point_to_line_dist(np.array([x, y]), line) + 1
            #i#f distance == 0:
            #    print("hm", line, y, x, distance)
            w = 255 / distance
            outline[y, x] = max(w, outline[y, x])
        #print(y)

for annotation in annotations:
    image_filename = annotation['filename']

    image = cv2.imread(os.path.join(data_path, image_filename))
    height, width = image.shape[:2]
    #cv2.imshow("im", image)
    #cv2.waitKey()

    outline = np.zeros((image.shape[0], image.shape[1]))
    
    polygons = [region['shape_attributes'] for region in annotation['regions']]
    for polygon in polygons:
        points_x, points_y = polygon['all_points_x'], polygon['all_points_y']
        line_count = len(points_x) - 1
        lines = np.zeros((line_count, 2, 2))
        for line_idx in range(line_count):
            lines[line_idx] = [[points_x[line_idx], points_y[line_idx]], [points_x[line_idx+1], points_y[line_idx+1]]]
        
        for line in lines:
            draw_outline(outline, line)

    outline = outline / 255.0
    cv2.imshow("im", outline)
    cv2.waitKey()







    print(image_filename)
    quit()


print(annotations)
quit()

