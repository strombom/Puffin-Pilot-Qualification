
import math
import numpy
from numba import jit, njit

@njit
def norm(v):
	return math.sqrt(v[0]*v[0] + v[1]*v[1])

@njit
def make_line_from_points(points, points_count):
	p1 = get_farthest_point_from_point(points, points_count, points[0])
	p2 = get_farthest_point_from_point(points, points_count, p1)
	return (p1, p2)

@njit
def distance_from_point_to_points(point, points, points_count):
    if points_count == 0:
        return 0.0

    min_distance = 1e9
    for pidx in range(points_count):
        v = point - points[pidx]
        min_distance = min(min_distance, norm(v))

    return min_distance

@njit
def get_farthest_point_from_point(points, points_count, p):
    farthest_pidx = 0
    farthest_distance = 0
    for pidx in range(points_count):

        # No need for exact distance, a relative distance is ok
        v = points[pidx] - p
        distance = v[0]*v[0] + v[1]*v[1]

        if distance > farthest_distance:
            farthest_distance = distance
            farthest_pidx = pidx

    return points[farthest_pidx]

@njit
def get_farthest_point_from_line(points, points_count, p1, p2):
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
