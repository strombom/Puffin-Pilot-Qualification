
import math
import numpy as np
from numba import jit, njit

@njit
def norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

@njit
def line_length(line):
    return norm(line[1] - line[0])

@njit
def make_line_from_points(points, points_count):
    line_1 = np.empty((2, 2))
    line_2 = np.empty((2, 2))
    line_3 = np.empty((2, 2))

    line_1[0] = get_point_farthest_from_points(points, points_count, points[0])
    line_1[1] = get_point_farthest_from_points(points, points_count, line_1[0])

    if points_count > 3:
        p2 = get_closest_point(points, points_count, line_1[0])
        p3 = get_closest_point(points, points_count, line_1[1])

        line_2[0], line_2[1] = p2, line_1[1]
        line_3[0], line_3[1] = line_1[0], p3

        line_score_1 = line_score(line_1, points[1:points_count-1])
        line_score_2 = line_score(line_2, points[2:points_count-1])
        line_score_3 = line_score(line_3, points[1:points_count-2])

        if line_score_1 < line_score_2 and line_score_1 < line_score_3:
            return line_1
        elif line_score_2 < line_score_1 and line_score_2 < line_score_3:
            return line_2
        else:
            return line_3

    return line_1

@njit
def line_score(line, points):
    score = 0
    for idx in range(points.shape[0]):
        score += line_to_point_distance(line, points[idx]) ** 2
    return score

@njit
def has_common_point(points1, points1_count, points2, points2_count):
    for first_idx in range(points1_count):
        for second_idx in range(points2_count):
            if np.all(np.equal(points1[first_idx], points2[second_idx])):
                return True
    return False

@njit
def distance_from_points_to_point(points, points_count, point):
    if points_count == 0:
        return 0.0

    min_distance = 1e9
    for pidx in range(points_count):
        v = point - points[pidx]
        min_distance = min(min_distance, norm(v))

    return min_distance

@njit
def get_closest_point(points, points_count, point):
    min_pidx = 0
    min_distance = 1e9
    for pidx in range(points_count):
        # No need for exact distance, a relative distance is ok
        v = point - points[pidx]
        distance = v[0]*v[0] + v[1]*v[1]

        if distance > 1 and distance < min_distance:
            min_distance = distance
            min_pidx = pidx

    return points[min_pidx]

@njit
def get_point_farthest_from_points(points, points_count, point):
    farthest_pidx = 0
    farthest_distance = 0
    for pidx in range(points_count):
        # No need for exact distance, a relative distance is ok
        v = points[pidx] - point
        distance = v[0]*v[0] + v[1]*v[1]

        if distance > farthest_distance:
            farthest_distance = distance
            farthest_pidx = pidx

    return points[farthest_pidx]

@njit
def get_point_farthest_from_line(points, points_count, p1, p2):
    # Calculate approximately shortest distance from 
    #  line p1-p2 to all cluster points
    minimum_distance = 4.0

    # Pre calculate the line direction
    v0 = (p2 - p1) / norm(p2 - p1)

    farthest_distance = 0
    farthest_pidx = 0
    for pidx in range(points_count):
        p3 = points[pidx]
        v1 = p3 - p1
        v1_magn = norm(v1)

        # Make sure that the point is not too close to our reference points
        if v1_magn < minimum_distance:
            continue

        line_point_distance = abs(norm(v1 / v1_magn - v0) * v1_magn)
        if line_point_distance > farthest_distance:
            farthest_distance = line_point_distance
            farthest_pidx = pidx

    return points[farthest_pidx]

@njit
def get_points_close_to_line(cluster, line, line_points, line_points_count):
    max_distance = 4.0

    # Calculate line vector
    p1, p2 = line[0], line[1]
    v0 = (p2 - p1) / norm(p2 - p1)

    for p_idx in range(cluster.points_count):
        # Calculate point vector
        v1 = cluster.points[p_idx] - p1
        v1_magn = norm(v1)

        append_point = False
        if v1_magn < max_distance:
            # The point is very close to the start point.
            append_point = True
        else:
            line_point_distance = norm(v1 / v1_magn - v0) * v1_magn
            if line_point_distance < max_distance:
                # The point is very close to the line.
                append_point = True

        if append_point:
            line_points[line_points_count] = cluster.points[p_idx]
            line_points_count[0] += 1

@njit
def line_to_line_angle(l1, l2):
    v1, v2 = l1[1] - l1[0],  l2[1] - l2[0]
    #v1, v2 = v1 / norm(v1), v2 / norm(v2)
    dot    = v1[0] * v2[0] + v1[1] * v2[1]
    det    = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)

@njit
def line_to_point_angle(line, point):
    v1, v2 = line[1] - line[0], point - line[0]
    v1, v2 = v1 / norm(v1), v2 / norm(v2)
    dot    = v1[0] * v2[0] + v1[1] * v2[1]
    det    = v1[0] * v2[1] - v1[1] * v2[0]
    angle  = abs(math.atan2(det, dot))
    if angle > math.pi / 2:
        angle = math.pi - angle
    return angle

@njit
def line_to_point_distance(line, point):
    distance = norm(point - line[0])
    v_line = line[1] - line[0]
    point_on_line = distance * v_line / norm(v_line) + line[0]
    return norm(point - point_on_line)

@njit
def point_to_point_distance(p1, p2):
    return norm(p2 - p1)
