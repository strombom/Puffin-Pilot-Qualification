
from __future__ import print_function

import math
import numpy as np
from numba import jit, njit, jitclass
from numba import int64, uint64, float64, boolean

line_distance_threshold = 2.5

line_spec = [
    ('angle',     float64),
    ('count',     uint64),
    ('_points',   float64[:,:]),
    ('_line',     uint64[:,:]),
    ('_has_line', boolean)
]

#@jitclass(line_spec)
class Line(object):
    def __init__(self, angle, count, points):
        self.angle = angle
        self.count = count
        self._points = points
        self._line = np.empty((2, 2))
        self._line_done = False

    def get_line(self):
        self._make_line()
        return self._line

    def get_points(self):
        self._make_line()
        return self._points

    def _make_line(self):
        if self._line_done:
            return
        self._line[0] = get_point_farthest_from_points(self._points, self.count, self._points[0])
        self._line[1] = get_point_farthest_from_points(self._points, self.count, self._line[0])
        distances = np.zeros(self.count)
        for idx in range(self.count):
            distances[idx] = distance_squared(self._line[0], self._points[idx])
        indices = np.argsort(distances)
        self._points = self._points[indices]
        self._line_done = True


    def get_fitness(self, ref_line):
        self._make_line()
        ref_line._make_line()

        print("get_fitness")

        # Angle diff, higher number is better, max pi/2 (perpendicular)
        angle_diff = abs(self.angle - ref_line.angle)
        if angle_diff > math.pi / 2.0:
            angle_diff = math.pi - angle_diff
        angle_fitness = angle_diff / 2.0

        # Distance between end points, smaller is better
        #  Normally between 5 and 30
        min_d = 1e9
        for i in range(2):
            for j in range(2):
                d = distance_squared(self._line[i], ref_line._line[j])
                min_d = min(min_d, d)
        min_d = min_d ** 0.5
        distance_fitness = 1 / (1 + min(30, min_d) / 30.0)

        # Line overlap
        p0 = line_intersection(self._line, ref_line._line)
        v0 = self._line[0] - p0
        v1 = self._line[1] - p0
        v0_len = norm(v0)        
        if v0_len == 0:
            overlap_fitness = 1
        else:
            l = line_length(self._line)
            pc = (v0 - v1) / 2
            p2 = pc - v0 / v0_len * l / 2
            distance = norm(p2)
            overlap_fitness = 1 / (1 + distance / 15.0)
            """
            print(self._line)
            print(ref_line._line)
            print("p0", p0)
            print("v0", v0)
            print("v1", v1)
            print("pc", pc)
            print("l", l)
            print("p2", p2)
            print("angle", angle_fitness)
            print("distance", distance_fitness)
            print("overlap", overlap_fitness)
            """



        return angle_fitness + distance_fitness + overlap_fitness

def line_repr(line):
    l = line.get_line()
    return "Line(%1.2f, % 2d, [[%1.2f %1.2f] [%1.2f %1.2f]])" % (line.angle, line.count, l[0][0], l[0][1], l[1][0], l[1][1])






"""
print("get ordered")
print("====")
line_size = 0

for idx in range(points_count):
if idx == best_line.i or idx == best_line.j:
    line_pair[line_size] = points[idx]
    line_size += 1
    continue
distance = line_to_point_distance(best_line.line, points[idx])
print(best_line.line)
print(distance, distance_threshold)
if distance < distance_threshold:
    line_pair[line_size] = points[idx]
    line_size += 1

#print(line_repr(line_pair))
print(points[0:points_count])
print(line_pair)

# Store sorted result and return length
line_pair[0:line_size] = sort_points(line_pair[0:line_size])
print("====")
return line_size
"""
"""
@njit
def get_ordered_line_points(points, points_count, best_line, distance_threshold, line_pair):
print("get ordered")
print("====")
line_size = 0




for idx in range(points_count):
    if idx == best_line.i or idx == best_line.j:
        line_pair[line_size] = points[idx]
        line_size += 1
        continue
    distance = line_to_point_distance(best_line.line, points[idx])
    print(best_line.line)
    print(distance, distance_threshold)
    if distance < distance_threshold:
        line_pair[line_size] = points[idx]
        line_size += 1

#print(line_repr(line_pair))
print(points[0:points_count])
print(line_pair)

# Store sorted result and return length
line_pair[0:line_size] = sort_points(line_pair[0:line_size])
print("====")
return line_size
"""


def find_all_lines(points, points_count, line_pair):
    # Find all lines with at least 3 points
    print("Find all lines")
    print(points)
    print("===")

    if points_count == 3:
        print("find lines with 3 points")
        quit()

    line_pair_size = jit_find_all_lines(points, points_count, line_pair)

    if line_pair_size[0] == 0:
        # No line found
        print("no line found")
        quit()

    if line_pair_size[1] == 0:
        print(line_pair_size)
        print(line_pair)
        # Only one line found
        print("one line found")
        quit()

    return line_pair_size


#@njit
def jit_find_all_lines(points, points_count, line_pair):
    print("points_count", points_count)
    line_pair_size = np.zeros(2, dtype=np.uint64)

    max_lines_count = 50
    lines_counts = np.zeros((max_lines_count),                  dtype=np.uint64)
    lines_angles = np.empty((max_lines_count),                  dtype=np.float64)
    lines_points = np.empty((max_lines_count, points_count, 2), dtype=np.float64)
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
                    count += 1
            p = line_ij[1] - line_ij[0]
            angle = math.atan2(p[1], p[0])
            if angle < 0:
                angle = math.pi + angle
            if count > 2:
                lines_angles[lines_count] = angle
                lines_counts[lines_count] = count
                lines_count += 1
            if lines_count == max_lines_count:
                break
        if lines_count == max_lines_count:
            break

    if lines_count < 3:
        # No first line found
        return line_pair_size

    # Find best line
    first_idx = np.argmax(lines_counts[0:lines_count])
    first_line = Line(angle  = lines_angles[first_idx],
                      count  = lines_counts[first_idx],
                      points = lines_points[first_idx])

    # Store the first line
    line_pair[0][0:first_line.count] = first_line.get_points()
    line_pair_size[0] = first_line.count
    #get_ordered_line_points(points, points_count, first_line, distance_threshold, line_pair[0])
    #print(line_pair[0][0:line_pair_size[0]])

    # Remove all lines with similar angle as the best line
    minimum_angle = math.radians(35)
    for idx in range(lines_count):
        angle_diff = abs(first_line.angle - lines_angles[idx])
        if angle_diff > math.pi / 2.0:
            angle_diff = math.pi - angle_diff
        if angle_diff < minimum_angle:
            lines_counts[idx] = 0

    # Sort lines by size
    lines_idc = np.argsort(lines_counts[0:lines_count])[::-1]
    lines_angles = lines_angles[lines_idc]
    lines_counts = lines_counts[lines_idc]
    lines_points = lines_points[lines_idc]
    second_line = None

    if lines_counts[0] == 0:
        # No second line found

        # Check if one point can be used as line
        best_fitness = 0
        
        first_line_points = first_line.get_points()
        print("second line", points_count)
        print(first_line_points)
        print("====")
        for k in range(points_count):
            #print("k", points[k])
            #print(first_line_points)
            #print(points)
            #print("--")
            inside = False
            for m in range(first_line_points.shape[0]):
                #print(first_line_points[m])
                if np.array_equal(first_line_points[m], points[k]):
                    inside = True
                    break
            if inside:
                continue
            #print("not inside", points[k])

            for l in range(2):


                line_ij[0] = points[k]
                if l == 0:
                    line_ij[1] = first_line.get_points()[0]
                else:
                    line_ij[1] = first_line.get_points()[-1]

                p = line_ij[1] - line_ij[0]
                if norm(p) < 4:
                    continue

                angle = math.atan2(p[1], p[0])
                if angle < 0:
                    angle = math.pi + angle
                angle_diff = abs(first_line.angle - angle)
                if angle_diff > math.pi / 2.0:
                    angle_diff = math.pi - angle_diff
                #print("angle_diff", angle_diff, line_ij[0], line_ij[1])
                if angle_diff < minimum_angle:
                    continue

                test_line = Line(angle  = angle,
                                 count  = 2,
                                 points = line_ij)

                fitness = test_line.get_fitness(first_line)
                #print("fitness", fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    second_line = test_line
                    #print("best fitness")


                #print("testline", line_repr(test_line))
                #print(".")

        #print("--")
        #print(first_line.get_points())

        # print(lines_counts[])

        #quit()

    elif lines_counts[0] == lines_counts[1]:
        # Many candidates with similar score, pick the best one
        best_fitness = 0
        idx = 0
        while lines_counts[idx] == lines_counts[0]:
            test_line = Line(angle  = lines_angles[idx],
                             count  = lines_counts[idx],
                             points = lines_points[idx])

            fitness = test_line.get_fitness(first_line)
            if fitness > best_fitness:
                best_fitness = fitness
                second_line = test_line
            idx += 1

    else:
        print("only one line candidate")
        quit()

    if second_line is not None:
        # Store the second line
        line_pair[1][0:second_line.count] = second_line.get_points()
        line_pair_size[1] = second_line.count

    return line_pair_size



#if angle_diff 
#print("a0 %1.2f %1.2f -> %1.2f" % (best_line.angle, lines_angles[idx], angle_diff))
#if angle_diff > minimum_angle:
#    c_line = Line(i     = lines_coords[idx][0],
#                  j     = lines_coords[idx][1],
#                  angle = lines_angles[idx],
#                  count = lines_counts[idx])
#    print("Candidate", line_repr(c_line))
#continue



#print(lines_angles)

#for idx in range(lines_count):
#    print("Line(%d, %d, %0.2f, %d)" % (lines_coords[idx][0], lines_coords[idx][1], lines_angles[idx], lines_counts[idx]))
#if line[0] == 0:
#    break
#print(line_repr(line))

#sorted(lines, key=line_compare, reverse = True)


"""
    quit()






    return best_count






    line = np.empty((2, 2))
    best_count = 0
    for i in range(points_count):
        for j in range(i+1, points_count):
            line[0], line[1] = points[i], points[j]
            count = 0
            for k in range(points_count):
                if k==i or k==j:
                    continue
                distance = line_to_point_distance(line, points[k])
                if distance < distance_threshold:
                    #print(distance)
                    count += 1
            if count > best_count:
                best_pair[:] = i, j
                best_count = count
    return best_count

"""

"""
    if points_count == 3:
        

    if points_count < 4:
        # If there are only 2 or 3 points, simply return a line
        #  consisting of the two extreme points.
        line[0] = get_point_farthest_from_points(points, points_count, points[0])
        line[1] = get_point_farthest_from_points(points, points_count, line[0])
        return line
"""

def line_finder(points, points_count, lines):







    distances = np.zeros(points_count, dtype=np.float64)
    for idx in range(points_count):
        distances[idx] = distance_squared(points[idx], first_point)
    indices = np.argsort(distances)
    ordered_points = points[0:points_count][indices]


    good_lines = np.empty((5, 2, 2))
    good_lines_count = 0
    for i in range(points_count - 1):
        line[0], line[1] = ordered_points[i], ordered_points[i+1]
        print("===")
        count = 0
        for j in range(points_count):
            if i==j or i+1==j:
                continue
            d = line_to_point_distance(line, ordered_points[j])
            print(d)

    print(ordered_points)

    quit()







    print("ransac")
    print(points)
    print(points_count)

    inliers_min = 4



    quit()












#@njit
def make_line_from_points(points, points_count):
    line = np.empty((2, 2))
    if points_count < 4:
        # If there are only 2 or 3 points, simply return a line
        #  consisting of the two extreme points.
        line[0] = get_point_farthest_from_points(points, points_count, points[0])
        line[1] = get_point_farthest_from_points(points, points_count, line[0])
        return line

    # Sort points by distance from one extreme point
    first_point = get_point_farthest_from_points(points, points_count, points[0])
    distances = np.zeros(points_count, dtype=np.float64)
    for idx in range(points_count):
        distances[idx] = distance_squared(points[idx], first_point)
    indices = np.argsort(distances)
    ordered_points = points[0:points_count][indices]

    if points[0][0] > 640 and points[0][1] > 550:
        if points_count > 6:
            print(ordered_points)

    residual_threshold = 1.0

    if points_count > 4:
        line[0], line[1] = ordered_points[0], ordered_points[-1]
        #print(ordered_points)
        residual_1 = line_score(line, ordered_points)
        line[0], line[1] = ordered_points[0], ordered_points[-2]
        #print(ordered_points[:-1])
        residual_2 = line_score(line, ordered_points[:-1])
        line[0], line[1] = ordered_points[1], ordered_points[-1]
        #print(ordered_points[1:])
        residual_3 = line_score(line, ordered_points[1:])

        if residual_1 < residual_threshold:
            line[0], line[1] = ordered_points[0], ordered_points[-1]
            return line

        elif residual_2 < residual_threshold and residual_2 < residual_3:
            line[0], line[1] = ordered_points[0], ordered_points[-2]
            return line

        elif residual_3 < residual_threshold:
            line[0], line[1] = ordered_points[1], ordered_points[-1]
            return line



        #print(residual_1, residual_2, residual_3)











    # Remove one point, try to fit a line to the remaining points.
    #  Do this for each point and chose the best fitting line.
    min_residual = 1e9
    min_idx = 0
    line_points = np.empty((points_count-1, 2))
    for i in range(points_count):
        line_points[0:i], line_points[i:] = ordered_points[0:i], ordered_points[i+1:]
        line[0], line[1] = line_points[0], line_points[-1]
        residual = line_score(line, line_points)
        if residual < min_residual:
            min_residual = residual
            min_idx = i
    line_points[0:min_idx], line_points[min_idx:] = ordered_points[0:min_idx], ordered_points[min_idx+1:]
    line[0], line[1] = line_points[0], line_points[-1]

    print(min_residual)


    return line

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

#@njit
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
