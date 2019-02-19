
from __future__ import print_function

import math
import numpy as np
from numba import jit, njit, jitclass
from numba import int64, uint64, float64, boolean

MAX_POINTS = 15

line_distance_threshold = 2.7

@njit
def get_line_fitness(line_angle, line_line, ref_line_angle, ref_line_line):
    # Angle diff, higher number is better, max pi/2 (perpendicular)
    angle_diff = abs(line_angle - ref_line_angle)
    if angle_diff > math.pi / 2.0:
        angle_diff = math.pi - angle_diff
    angle_fitness = angle_diff / 2.0

    # Distance between end points, smaller is better
    #  Normally between 5 and 30
    min_d = 1e9
    for i in range(2):
        for j in range(2):
            d = distance_squared(line_line[i], ref_line_line[j])
            min_d = min(min_d, d)
    min_d = min_d ** 0.5
    distance_fitness = 1 / (1 + min(30, min_d) / 30.0)

    # Line overlap
    p0 = np.array(line_intersection(line_line, ref_line_line))
    v0 = line_line[0] - p0
    v1 = line_line[1] - p0
    v0_len = norm(v0)        
    if v0_len == 0:
        overlap_fitness = 1
    else:
        l = line_length(line_line)
        pc = (v0 - v1) / 2
        p2 = pc - v0 / v0_len * l / 2
        distance = norm(p2)
        overlap_fitness = 1 / (1 + distance / 15.0)
    return angle_fitness + distance_fitness + overlap_fitness

@njit
def process_line(points, count, line):
    idx_0 = int(get_point_farthest_from_points(points, count, points[0]))
    line[0] = points[int(idx_0)]
    idx_1 = get_point_farthest_from_points(points, count, line[0])
    line[1] = points[int(idx_1)]

    # Sort by distance
    distances = np.zeros(MAX_POINTS, np.float64)
    idx = 0
    while idx < count:
        x1, y1 = line[0][0], line[0][1]
        x2, y2 = points[idx][0], points[idx][1]
        distance = (x1-x2)**2 + (y1-y2)**2
        distances[idx] = distance
        idx += 1
    indices = np.argsort(distances[0:count])
    points[0:count] = points[indices]

    # Find outliers
    if count > 3:
        d_tot = norm(points[0] - points[int(count-1)]) / (count - 1)
        d_0 = norm(points[0] - points[1])
        d_1 = norm(points[int(count-2)] - points[int(count-1)])

        if d_0 > d_tot * 1.2:
            count -= 1
            points[0:int(count)] = points[1:int(count+1)]
            #quit()
        elif d_1 > d_tot * 1.2:
            count -= 1

    idx_0 = get_point_farthest_from_points(points, count, points[0])
    line[0] = points[int(idx_0)]
    idx_1 = get_point_farthest_from_points(points, count, line[0])
    line[1] = points[int(idx_1)]

    return int(count)

@njit
def find_all_lines(points, points_count, line_pair, line_pair_size):
    max_lines_count = 50
    lines_counts    = np.zeros((max_lines_count),                dtype=np.uint64)
    lines_angles    = np.empty((max_lines_count),                dtype=np.float64)
    lines_points    = np.empty((max_lines_count, MAX_POINTS, 2), dtype=np.float64)
    lines_residuals = np.zeros((max_lines_count),                dtype=np.float64)
    lines_count = 0

    # Find all possible lines
    line_ij = np.empty((2, 2))
    for i in range(points_count):
        for j in range(i+1, points_count):
            line_ij[0], line_ij[1] = points[i], points[j]
            count = 0
            for k in range(points_count):
                if k==i or k==j:
                    lines_points[lines_count][count] = points[k]
                    count += 1
                    continue
                distance = line_to_point_distance(line_ij, points[k])
                if distance < line_distance_threshold:
                    lines_points[lines_count][count] = points[k]
                    lines_residuals[lines_count] += distance
                    count += 1
            p = line_ij[1] - line_ij[0]
            angle = math.atan2(p[1], p[0])
            if angle < 0:
                angle = math.pi + angle
            if count > 2:
                lines_angles[lines_count] = angle
                lines_counts[lines_count] = count
                lines_residuals[lines_count] /= (count-2)
                lines_count += 1
            if lines_count == max_lines_count:
                break
        if lines_count == max_lines_count:
            break

    if lines_count < 3:
        # No first line found
        return

    # Find best line
    first_idx = -1
    best_residual = 1e9
    #for idx in range(lines_count):
    #    if lines_counts[idx] > 3:
    #        if lines_residuals[idx] < best_residual:
    #            best_residual = lines_residuals[idx]
    #            first_idx = idx
    if first_idx == -1:
        # Use longest line
        first_idx = np.argmax(lines_counts[0:lines_count])
        if first_idx < 0:
            return

    first_line_angle = lines_angles[int(first_idx)]
    first_line_count = lines_counts[int(first_idx)]
    first_line_line = np.empty((2, 2), np.float64)    
    first_line_points = np.empty((MAX_POINTS, 2), np.float64)
    first_line_points[:] = lines_points[int(first_idx)]
    first_line_count = process_line(first_line_points, first_line_count, first_line_line)

    # Store the first line
    line_pair[0][0:int(first_line_count)] = first_line_points[0:int(first_line_count)]
    line_pair_size[0] = first_line_count
    #get_ordered_line_points(points, points_count, first_line, distance_threshold, line_pair[0])

    # Remove all lines with similar angle as the best line
    minimum_angle = math.radians(35)
    for idx in range(lines_count):
        angle_diff = abs(first_line_angle - lines_angles[idx])
        if angle_diff > math.pi / 2.0:
            angle_diff = math.pi - angle_diff
        if angle_diff < minimum_angle:
            lines_counts[idx] = 0

    # Sort lines by size
    lines_idc = np.argsort(lines_counts[0:lines_count])[::-1]
    lines_angles = lines_angles[lines_idc]
    lines_counts = lines_counts[lines_idc]
    lines_points = lines_points[lines_idc]

    second_line_angle =0.0
    second_line_count = 0
    second_line_line = np.empty((2, 2), np.float64)    
    second_line_points = np.empty((MAX_POINTS, 2), np.float64)

    if lines_counts[0] == 0:
        # No second line found

        # Check if one point can be used as line
        best_fitness = 0
        
        #first_line_points = first_line.get_points()
        for k in range(points_count):
            inside = False
            for m in range(first_line_count):
                if first_line_points[m][0] == points[k][0] and \
                   first_line_points[m][1] == points[k][1]:
                    inside = True
                    break
            if inside:
                continue

            for l in range(2):
                line_ij[0] = points[k]
                if l == 0:
                    line_ij[1] = first_line_points[0]
                else:
                    line_ij[1] = first_line_points[int(first_line_count-1)]

                p = line_ij[1] - line_ij[0]
                if norm(p) < 4:
                    continue

                angle = math.atan2(p[1], p[0])
                if angle < 0:
                    angle = math.pi + angle
                angle_diff = abs(first_line_angle - angle)
                if angle_diff > math.pi / 2.0:
                    angle_diff = math.pi - angle_diff
                if angle_diff < minimum_angle:
                    continue

                second_line_angle = angle
                second_line_count = 2
                second_line_points[0:2] = line_ij
                second_line_line[:] = line_ij

                fitness = get_line_fitness(second_line_angle, second_line_line, first_line_angle, first_line_line)

                if fitness > best_fitness:
                    best_fitness = fitness
                    line_pair[1][0:second_line_count] = second_line_points[0:second_line_count]
                    line_pair_size[1] = second_line_count


    else:
        best_fitness = 0
        best_idx = -1
        idx = 0
        while lines_counts[idx] == lines_counts[0]:
            second_line_angle     = lines_angles[idx]
            second_line_count     = lines_counts[idx]
            second_line_points[:] = lines_points[idx]
            second_line_count = process_line(second_line_points, second_line_count, second_line_line)

            fitness = get_line_fitness(second_line_angle, second_line_line, first_line_angle, first_line_line)
            if fitness > best_fitness:
                best_fitness = fitness
                line_pair[1][0:int(second_line_count)] = second_line_points[0:int(second_line_count)]
                line_pair_size[1] = second_line_count
                best_idx = idx
            idx += 1

    return


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

    return farthest_pidx

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
    max_distance = 114.0

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
    # Cross product
    a = (line[1][0] - line[0][0]) * (line[0][1] -   point[1])
    b = (line[0][0] -   point[0]) * (line[1][1] - line[0][1])
    # Length
    d = (line[1][0] - line[0][0]) ** 2
    e = (line[1][1] - line[0][1]) ** 2
    return abs(a - b) / math.sqrt(d + e)

@njit
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

@njit
def norm(v):
    return math.sqrt(v[0]*v[0] + v[1]*v[1])

@njit
def point_to_point_distance(p1, p2):
    return norm(p2 - p1)

@njit
def line_length(line):
    return norm(line[1] - line[0])

@njit
def distance_squared(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

@njit
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
    x =  det(d, xdiff) / div
    y =  det(d, ydiff) / div
    return x, y
