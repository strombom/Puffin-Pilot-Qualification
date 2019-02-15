
import cv2
import skimage
import numpy as np
import skimage.morphology
from numba import jit, njit


fid_size      = 2      # Fiducial max radius extent
min_area      = 14     # Minimum acceptable fiducial area
img_threshold = 0.996

def extract_points(image, max_points):
    points = np.empty((max_points, 2))
    point_count = jit_extract_points(image, points)
    return points[0:point_count]

@njit
def get_section(image, x, y):
    return image[y-fid_size:y+fid_size+1,x-fid_size:x+fid_size+1]

@njit
def get_section_area(image, x, y):
    return np.sum(image[y-fid_size:y+fid_size+1,x-fid_size:x+fid_size+1]) # / 256

@njit
def get_center_of_gravity(image, px, py):
    section = get_section(image, px, py)

    my, mx = 0.0, 0.0
    count = 0
    for y in range(section.shape[0]):
        for x in range(section.shape[1]):
            if section[y][x] > img_threshold:
                my += y
                mx += x
                count += 1
    
    if count < min_area:
        return 0, 0

    return mx / count, my / count

@njit
def jit_extract_points(image, points):
    point_count = 0
    min_x, max_x = fid_size, image.shape[1] - 1 - fid_size
    min_y, max_y = fid_size, image.shape[0] - 1 - fid_size

    print("minmax", np.max(image), np.min(image))

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y][x] < img_threshold:
                continue
            section = None

            # Keep the section within the image
            x = min(max(x + 1, min_x), max_x)
            y = min(max(y, min_y), max_y)

            # Find section that ideally covers the entire point
            px, py = x, y
            section_area = get_section_area(image, px, py)
            for step in range(fid_size):
                if py + 1 + fid_size == image.shape[0]:
                    break
                new_x = 0
                for dx in (0, 1, -1):
                    if px + dx < fid_size or px + dx + fid_size >= image.shape[1]:
                        continue
                    # Can be sped up but there are normally only 40 points so not worth it
                    new_section_area = get_section_area(image, px + dx, py + 1)
                    if new_section_area > section_area:
                        section_area = new_section_area
                        new_x = px + dx
                if new_x == 0:
                    break
                px = new_x
                py += 1

            # Get center of point if found
            mx, my = get_center_of_gravity(image, px, py)
            if mx > 0:
                cx, cy = px + mx - fid_size, py + my - fid_size
                points[point_count] = (cx, cy)
                point_count += 1
                if point_count == points.shape[0]:
                    return point_count

                # Erase found point from image
                section = get_section(image, px, py)
                section[:] = 0.5

    return point_count




"""
    import time

    max_points = 10000
    points = np.empty((max_points, 2))

    for i in range(20):
        im2 = image.copy()
        t = time.time()
        point_count = _extract_points(im2, points)
        print(time.time() - t)

    for point in points:
        x = int(point[0])
        y = int(point[1])
        im2[y-1][x] = 0
        im2[y][x-1] = 200
        im2[y][x] = 200
        im2[y][x+1] = 200
        im2[y+1][x] = 200

    cv2.imwrite("out.png", im2)
"""

