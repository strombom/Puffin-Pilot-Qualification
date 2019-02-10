
import numpy as np


def quad_to_poly(quadrilateral, section_count = 10):
    polygon = np.empty((section_count * 4, 1, 3))
    line = np.empty((2, 3))

    for line_idx in range(4):
        start_idx, stop_idx = line_idx, (line_idx + 1) % 4
        line_start = quadrilateral[start_idx,0,:]
        line_end   = quadrilateral[stop_idx, 0,:]

        direction   = normalize(line_end - line_start)
        section_len = magnitude(line_end - line_start) / section_count

        for section_idx in range(section_count):
            polygon[line_idx * section_count + section_idx,0] = line_start + direction * section_len * section_idx
    
    return polygon

def sort_corners(corners):
    # Sort corners clockwise
    corners = corners[corners[:,0].argsort()]
    c_left, c_right = corners[0:2], corners[2:4]
    c_left  = c_left [c_left [:,1].argsort()]
    c_right = c_right[c_right[:,1].argsort()]
    corners[:] = [c_left [0], c_right[0], c_right[1], c_left [1]]
    return corners

def rotate_points(points, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s, 0), (s, c, 0), (0, 0, 0)))
    return points.dot(R)

def make_corners(p, z = 0.0):
    corners = np.empty((4, 1, 3))
    for i in range(4):
        corners[i][0] = rotate_points(np.array([(-p, -p, z)]), -i * np.radians(90))[0]
    return corners

def normalize(v):
    return v / np.linalg.norm(v)

def magnitude(v):
    return np.linalg.norm(v)
