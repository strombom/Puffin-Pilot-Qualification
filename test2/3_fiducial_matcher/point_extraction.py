
import cv2
import skimage
import numpy as np
import skimage.morphology
from numba import jit, njit


fid_size  = 3  # Fiducial max radius extent
min_area  = 16 # Minimum acceptable fiducial area


def extract_points(img, max_points):
    points = np.empty((max_points, 2))
    point_count = jit_extract_points(img, points)
    return points[0:point_count]


@njit
def get_section(img, x, y):
    return img[y-fid_size:y+fid_size+1,x-fid_size:x+fid_size+1]

@njit
def get_section_area(img, x, y):
    return np.sum(img[y-fid_size:y+fid_size+1,x-fid_size:x+fid_size+1]) / 256

@njit
def get_center_of_gravity(img, px, py):
    section = get_section(img, px, py)

    my, mx = 0.0, 0.0
    count = 0
    for y in range(section.shape[0]):
        for x in range(section.shape[1]):
            if section[y][x] != 0:
                my += y
                mx += x
                count += 1
    
    if count < min_area:
        return 0, 0

    return mx / count, my / count

@njit
def jit_extract_points(img, points):
    point_count = 0
    min_x, max_x = fid_size, img.shape[1] - 1 - fid_size
    min_y, max_y = fid_size, img.shape[0] - 1 - fid_size

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] < 253:
                continue
            section = None

            # Keep the section within the image
            x = min(max(x + 1, min_x), max_x)
            y = min(max(y, min_y), max_y)

            # Find section that ideally covers the entire point
            px, py = x, y
            section_area = get_section_area(img, px, py)
            for step in range(fid_size):
                if py + 1 + fid_size == img.shape[0]:
                    break
                new_x = 0
                for dx in (0, 1, -1):
                    if px + dx < fid_size or px + dx + fid_size >= img.shape[1]:
                        continue
                    # Can be sped up but there are normally only 40 points so not worth it
                    new_section_area = get_section_area(img, px + dx, py + 1)
                    if new_section_area > section_area:
                        section_area = new_section_area
                        new_x = px + dx
                if new_x == 0:
                    break
                px = new_x
                py += 1

            # Get center of point if found
            mx, my = get_center_of_gravity(img, px, py)
            if mx > 0:
                cx, cy = px + mx - fid_size, py + my - fid_size
                points[point_count] = (cx, cy)
                point_count += 1
                if point_count == points.shape[0]:
                    return point_count

                # Erase found point from image
                section = get_section(img, px, py)
                section[:] = 0

    return point_count




"""
    import time

    max_points = 10000
    points = np.empty((max_points, 2))

    for i in range(20):
        im2 = img.copy()
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

