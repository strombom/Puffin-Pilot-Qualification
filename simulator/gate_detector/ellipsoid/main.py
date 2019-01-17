
import copy
import math
import skimage
import skimage.morphology
import numpy as np
import cv2
from scipy import stats
from numba import jit, njit, jitclass, int32, float32, float64


ellipse_fitting_threshold = 3.0


def read_image(filename):
    image = skimage.io.imread(filename)

    #thresh = skimage.filters.threshold_minimum(image, nbins=256, max_iter=10000)
    thresh = 180
    image = image > thresh
    image = skimage.morphology.skeletonize(image)

    # Remove edge pixels
    rr, cc = skimage.draw.line(0, 0, 0, image.shape[1]-1)
    image[rr, cc] = 0
    rr, cc = skimage.draw.line(image.shape[0]-1, 0, image.shape[0]-1, image.shape[1]-1)
    image[rr, cc] = 0
    rr, cc = skimage.draw.line(0, 0, image.shape[0]-1, 0)
    image[rr, cc] = 0
    rr, cc = skimage.draw.line(0, image.shape[1]-1, image.shape[0]-1, image.shape[1]-1)
    image[rr, cc] = 0

    #skimage.io.imsave("img_skel.png", image*255)
    #quit()

    return image

@jit(nopython=True)
def extract_strip(image, ox, oy, strip):
    tmp_strip = np.empty_like(strip)

    idx = 0
    is_first_direction = True
    is_first_point = True
    is_restarting = False
    first_direction_length = 0
    x, y = ox, oy

    while True:
        if idx == strip.shape[0]:
            return idx

        if is_restarting:
            x, y = ox, oy
            is_restarting = False
        else:
            # Improvement: Linear regression instead of
            #              simply using the end points.
            tmp_strip[idx] = (x, y)
            idx += 1
            image[y][x] = 0
        
        neighbours = image[y-1:y+2, x-1:x+2]
        neighbour_count = np.sum(neighbours)

        def get_neighbour():
            for n_idx in range(9):
                dx, dy = n_idx % 3, n_idx // 3
                if neighbours[dy][dx]:
                    return dx - 1, dy - 1

        if neighbour_count == 0:
            if is_first_direction:
                is_first_direction = False
                is_restarting = True
                first_direction_length = idx
                continue
            else:
                # End of strip
                break

        elif neighbour_count == 1:
            dx, dy = get_neighbour()
            x, y = x + dx, y + dy
            continue
        
        else:
            if is_first_point:
                is_first_point = False
                dx, dy = get_neighbour()
                x, y = x + dx, y + dy
                continue
            else:
                if is_first_direction:
                    is_first_direction = False
                    is_restarting = True
                    first_direction_length = idx
                    continue
                else:
                    # End of strip
                    break

    # Copy strip in order 
    second_direction_length = idx - first_direction_length
    strip[0:second_direction_length] = tmp_strip[first_direction_length:idx][::-1]
    strip[second_direction_length:idx] = tmp_strip[0:first_direction_length]

    return idx

@jit(nopython=True)
def find_next_strip(image, start_x, start_y):
    image_size = image.shape[0] * image.shape[1]
    image_width = image.shape[1]
    start_pos = start_x + start_y * image_width
    start_pos += 1

    for idx in range(start_pos, image_size):
        y, x = idx // image_width, idx % image_width
        if image[y,x] == 1:
            return y, x
    return 0, 0

@jit(nopython=False)
def extract_strips(image):
    max_strip_length = 1000
    minimum_strip_length = 6
    strips = []
    x, y = 1, 1
    while True:
        y, x = find_next_strip(image, x, y)
        if y == 0 and x == 0:
            break
        strip = np.zeros((max_strip_length, 2), dtype=np.int)
        strip_len = extract_strip(image, x, y, strip)
        if strip_len > minimum_strip_length:
            strips.append(strip)
    return strips

@jit(nopython=True)
def magn(v):
    return (v[0] * v[0] + v[1] * v[1]) ** 0.5

@jit(nopython=True)
def split_strips(strips):
    line_segments = np.zeros((1000,2,2), dtype=np.int32)

    segment_count = 0
    minimum_segment_length = 10
    maximum_point_to_line_distance = 1.3

    for strip in strips:
        # Get strip length
        strip_length = strip.shape[0]
        for idx in range(strip.shape[0]):
            if strip[idx][0] == 0:
                strip_length = idx
                break

        # Initialize line segment
        line_start_idx = 0
        l_dir = np.zeros(2, dtype=np.float32)
        p_dir = np.zeros(2, dtype=np.float32)

        for idx in range(1, strip_length):
            if segment_count == line_segments.shape[0]:
                return line_segments[0:segment_count]

            segment_length = idx - line_start_idx

            # Last point
            if idx == strip_length - 1:
                if segment_length > minimum_segment_length:
                    line_segments[segment_count][0] = strip[line_start_idx]
                    line_segments[segment_count][1] = strip[idx]
                    segment_count += 1
                break

            # Calculate line direction
            p_dir[:] = strip[idx] - strip[line_start_idx]
            p_dir /= magn(p_dir)
            l_dir[:] = (l_dir * segment_length + p_dir) / (segment_length + 1)
            l_dir /= magn(l_dir)

            # Check if current point is outside of the line
            distance = magn(p_dir - l_dir) * segment_length
            if distance > maximum_point_to_line_distance:
                # Store line segment
                line_segments[segment_count][0] = strip[line_start_idx]
                line_segments[segment_count][1] = strip[idx]
                segment_count += 1

                # Initialize next line segment
                line_start_idx = idx
                l_dir[:] = (0, 0)

    return line_segments[0:segment_count]

@jit(nopython=True)
def min_distance(l1, l2):
    # Find the two closest points in the line pair
    dmin = 99999
    for i in range(2):
        for j in range(2):
            d = np.sum((l1[i] - l2[j]) ** 2) ** 0.5
            if d < dmin:
                dmin = d
                min_1 = i
                min_2 = j

    # Make sure the closest points are the first points in each line segment
    if min_1 == 1:
        l1[:] = l1[::-1]
    if min_2 == 1:
        l2[:] = l2[::-1]

    return dmin

@jit(nopython=True)
def line_intersection_angle(l1, l2):
    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)

@jit(nopython=True)
def vector_intersection_angle(v1, v2):
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    det = v1[0] * v2[1] - v1[1] * v2[0]
    return math.atan2(det, dot)

@jit(nopython=True)
def line_merging_degree(d, theta, l1, l2):
    return 1 / (d*2 + 1) * math.cos(math.pi - theta) * min(l1, l2) / max(l1, l2)

@jit(nopython=True)
def line_length(line):
    return magn(line[1] - line[0])

@jit(nopython=True)
def make_line_pair_clockwise(line_pair):
    v1 = line_pair[0][1] - line_pair[0][0]
    v2 = line_pair[1][1] - line_pair[1][0]
    theta1 = math.atan2(v1[1], v1[0])
    theta2 = math.atan2(v2[1], v2[0])
    delta_theta = theta2 - theta1
    if delta_theta < 0:
        delta_theta += math.pi * 2
    if delta_theta < math.pi:
        # Swap line positions
        line_pair[:] = line_pair[::-1]
    # Change i direction
    line_pair[0][:] = line_pair[0][::-1]

@jit(nopython=True)
def form_line_pairs(line_segments):
    d_max = 5.0
    theta_min = 145.0 / 180 * math.pi
    theta_max = 179.9 / 180 * math.pi
    line_pairs = np.zeros((1000, 2, 2, 2))
    merging_degrees = np.zeros((1000,))
    pair_count = 0
    unused_count = 0

    for i in range(line_segments.shape[0]):
        for j in range(i + 1, line_segments.shape[0]):
            line_pairs[pair_count][0] = line_segments[i]
            line_pairs[pair_count][1] = line_segments[j]
            d = min_distance(line_pairs[pair_count][0], line_pairs[pair_count][1])
            theta = abs(line_intersection_angle(line_pairs[pair_count][0], line_pairs[pair_count][1]))
            if d < d_max and theta > theta_min: # and theta < theta_max:
                make_line_pair_clockwise(line_pairs[pair_count])
                li, lj = line_length(line_pairs[pair_count][0]), line_length(line_pairs[pair_count][1])
                merging_degrees[pair_count] = line_merging_degree(d, theta, li, lj)
                pair_count += 1
                if pair_count == merging_degrees.shape[0]:
                    return line_pairs, merging_degrees

                if theta > 176 / 180 * math.pi:
                    # Nearly 180 degrees, save rotated copy also
                    line_pairs[pair_count][0][0] = line_pairs[pair_count-1][1][1]
                    line_pairs[pair_count][0][1] = line_pairs[pair_count-1][0][1]
                    line_pairs[pair_count][1][0] = line_pairs[pair_count-1][1][0]
                    line_pairs[pair_count][1][1] = line_pairs[pair_count-1][0][0]
                    merging_degrees[pair_count] = merging_degrees[pair_count-1]
                    pair_count += 1
                    if pair_count == merging_degrees.shape[0]:
                        return line_pairs, merging_degrees

    return line_pairs[0:pair_count], merging_degrees[0:pair_count]

@jit(nopython=True)
def line_intersection(line1, line2):
    xdiff = np.array((line1[0][0] - line1[1][0], line2[0][0] - line2[1][0]))
    ydiff = np.array((line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]))
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    # No 180 degree line pairs should exist due to theta_max in form_line_pairs
    # so no need to check if div is zero.
    #if div == 0:
    #   raise Exception('no intersection')
    d = np.array((det(line1[0], line1[1]), det(line2[0], line2[1])))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array((x, y))

@jit(nopython=True)
def test_line_merging_condition(arc_line, line):
    aj, bj, ak, bk = line[0], line[1], arc_line[1], arc_line[0]
    theta = abs(line_intersection_angle((aj, bj), (ak, bk)))
    if theta < math.pi / 2:
        # Angle condition failed, check length condition
        m = line_intersection((aj, bj), (ak, bk))
        length = magn(aj - bj) + magn(ak - bk)
        length /= magn(m - bj) + magn(m - bk)
        if length > 0.7:
            # Length condition also failed
            return False
    return True

@jit(nopython=True)
def rotate_vector_2d(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    return np.dot(r, v)

ellipse_spec = [
    ('center', float64[:]),
    ('axes', float64[:]),
    ('angle', float64),
    ('area', float64),
    ('perimeter', float64)
]

@jitclass(ellipse_spec)
class Ellipse(object):
    def __init__(self, ellipse):
        self.center = np.array((ellipse[0][0], ellipse[0][1]), dtype=np.float64)
        self.axes = np.array((ellipse[1][0], ellipse[1][1]), dtype=np.float64) / 2
        self.angle = np.radians(ellipse[2])
        self.area = self._area()
        self.perimeter = self._perimeter()

    def is_inside(self, point, scale):
        p1 = (point - self.center) / scale
        p2 = rotate_vector_2d(p1, -self.angle)
        if p2[0]**2 / self.axes[0]**2 + p2[1]**2 / self.axes[1]**2 <= 1:
            return True
        else:
            return False

    def _area(self):
        return math.pi * self.axes[0] * self.axes[1]

    def _perimeter(self):
        a, b = self.axes
        return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

    def point_distance(self, p):
        # https://github.com/0xfaded/ellipse_demo/issues/1
        p = np.abs(rotate_vector_2d(p - self.center, -self.angle))
        tx, ty = 0.707106, 0.707106
        a, b = self.axes
        for itn in range(0, 2): # Originally 3, but 2 might be enough
            x, y = a * tx, b * ty
            ex = (a * a - b * b) * tx ** 3 / a
            ey = (b * b - a * a) * ty ** 3 / b
            rx, ry = x - ex, y - ey
            qx, qy = p[0] - ex, p[1] - ey
            r, q = np.hypot(ry, rx), np.hypot(qy, qx)
            tx = min(1, max(0, (qx * r / q + ex) / a))
            ty = min(1, max(0, (qy * r / q + ey) / b))
            t = np.hypot(tx, ty)
            tx, ty = tx / t, ty / t
        p2 = np.array(((a * tx), (b * ty)))
        return magn(p - p2)

    def fitting_score(self, points):
        score = 0
        for idx in range(points.shape[0]):
            d = self.point_distance(points[idx])
            if d <  ellipse_fitting_threshold:
                score += 1
        return score / points.shape[0]

    def to_tuple(self):
        return ((self.center[0], self.center[1]), (self.axes[0]*2, self.axes[1]*2), np.degrees(self.angle))

    def line_intersection(slope, inception):
        print("line_intersection")
        print("slope, inception", slope, inception)

        quit()

        #p0, p1 = ellipse.line_intersection(goal_end)


def test_fitting_condition(arcs, lines = None):
    arc_points = []
    for arc in arcs:
        for line in arc:
            arc_points.append(line[0])
            arc_points.append(line[1])
    arc_points = np.unique(arc_points, axis=0).astype(np.float64)

    line_points = []
    if lines is not None:
        for line in lines:
            line_points.append(line[0])
            line_points.append(line[1])
    if len(line_points) > 0:
        line_points = np.unique(line_points, axis=0)
        points = np.vstack((arc_points, line_points))
        points = np.unique(points, axis=0)
    else:
        points = arc_points

    if len(points) < 6:
        return True, None

    points = np.array(points).astype(dtype=np.int32)
    ellipse = Ellipse(cv2.fitEllipseDirect(points))

    score_arc = ellipse.fitting_score(arc_points)
    if len(line_points) > 0:
        score_line = ellipse.fitting_score(line_points)
    else:
        score_line = 1.0

    threshold_fit = 0.8

    if score_arc < threshold_fit or score_line < threshold_fit:
        return False, ellipse

    return True, ellipse

def merge_line_pairs(arcs, line_pairs, merging_degrees):

    for line_pair, merging_degree in zip(line_pairs, merging_degrees):
        line_i, line_j = line_pair

        # Check if lines are in any arc
        arc_idx_line_i, arc_idx_line_j = -1, -1
        for arc_idx, arc in enumerate(arcs):
            if np.array_equal(arc[0], line_j):
                arc_idx_line_j = arc_idx
            if np.array_equal(arc[-1], line_i):
                arc_idx_line_i = arc_idx

        if arc_idx_line_i >= 0 and arc_idx_line_j >= 0:
            arc_i = arcs[arc_idx_line_i]
            arc_j = arcs[arc_idx_line_j]

            if test_line_merging_condition(line_i, line_j):
                if test_fitting_condition([arc_i, arc_j], [])[0]:
                    arcs[arc_idx_line_i] = arc_i + arc_j
                    del arcs[arc_idx_line_j]

        elif arc_idx_line_i >= 0:
            # Join line j into arc
            arc_i = arcs[arc_idx_line_i]

            if test_line_merging_condition(arc_i[-2], line_j):
                if test_fitting_condition([arc_i], [line_j])[0]:
                    arc_i.append(line_j)

        elif arc_idx_line_j >= 0:
            # Join line i into arc
            arc_j = arcs[arc_idx_line_j]

            if test_line_merging_condition(arc_j[0], line_i):
                if test_fitting_condition([arc_j], [line_j])[0]:
                    arc_j.insert(0, line_i)

        else:
            # Create new arc
            arc = [line_i, line_j]
            arcs.append(arc)


def merge_line_segments(line_segments):
    # Form the set LPij and sort them according to merging degree
    line_pairs, merging_degrees = form_line_pairs(line_segments)
    sorted_pairs = sorted(zip(line_pairs, merging_degrees), key=lambda i: i[1], reverse=True)
    line_pairs, merging_degrees = zip(*sorted_pairs)
    line_pairs, merging_degrees = list(line_pairs), list(merging_degrees)

    # Group into good and bad
    merging_threshold, merging_idx = 0.5, 0
    line_pairs_good, merging_degrees_good = line_pairs, merging_degrees
    line_pairs_bad, merging_degrees_bad = [], []
    for idx, merging_degree in enumerate(merging_degrees):
        if merging_degree < merging_threshold:
            line_pairs_good, line_pairs_bad = line_pairs_good[0:idx], line_pairs_good[idx:]
            merging_degrees_good, merging_degrees_bad = merging_degrees_good[0:idx], merging_degrees_good[idx:]
            break

    arcs = []
    merge_line_pairs(arcs, line_pairs_good, merging_degrees_good)
    merge_line_pairs(arcs, line_pairs_bad, merging_degrees_bad)
    
    arcs_to_remove = [i for i, val in enumerate(arcs) if len(val) <= 3]
    for idx in reversed(arcs_to_remove):
        del arcs[idx]

    unused_line_segments = []
    for line_segment in line_segments:
        found = False
        for arc in arcs:
            for line in arc:
                if np.array_equal(line_segment, line) or np.array_equal(line_segment, line[::-1]):
                    found = True
                    break
            if found:
                break
        if not found:
            unused_line_segments.append(line_segment)

    return arcs, unused_line_segments

def form_arc_pairs(arcs):
    arc_pairs = []

    # Calculate rotating ranges
    for arc_idx, arc in enumerate(arcs):
        v0 = arc[0][1] - arc[0][0]
        v1 = arc[-1][1] - arc[-1][0]
        rotating_degree = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
        rotating_range = np.array((np.math.atan2(-v0[1], v0[0]) + math.pi,
                                   np.math.atan2(-v1[1], v1[0]) + math.pi))
        arcs[arc_idx] = (arc, rotating_degree, rotating_range)

    # Evaluate arc pairs
    for arc_idx_i in range(len(arcs)):
        for arc_idx_j in range(arc_idx_i + 1, len(arcs)):
            arc_i, degree_i, range_a = arcs[arc_idx_i]
            arc_j, degree_j, range_b = arcs[arc_idx_j]

            # Check that the arcs' angles do not overlap
            start_angle = min(range_a[1], range_b[1])            
            range_a, range_b = range_a - start_angle, range_b - start_angle

            if range_a[1] > range_b[1]:
                range_a, range_b = range_b, range_a

            # A small overlap is permitted
            max_coinciding_degree = min(degree_i, degree_j) / 10
            limit_0 = math.pi * 2 + max_coinciding_degree
            limit_1 = range_a[0] - max_coinciding_degree

            if range_b[0] > limit_0 or range_b[1] < limit_1:
                # Overlapping too much
                continue

            # Check rotating direction
            line_1 = np.array((arc_i[-1][1], arc_j[0][0]))
            line_2 = np.array((arc_j[-1][1], arc_i[0][0]))
            a = line_intersection_angle(arc_i[-1], line_1)
            b = line_intersection_angle(line_1, arc_j[0])
            c = line_intersection_angle(arc_j[-1], line_2)
            d = line_intersection_angle(line_2, arc_i[0])
            if a < 0 or b < 0 or c < 0 or d < 0:
                # At least one intersection has the wrong direction
                continue
            
            # Check fitting condition
            fitting_contition, ellipse = test_fitting_condition([arc_i, arc_j], None)
            if fitting_contition:
                arc_pairs.append(((arc_idx_i, arc_idx_j), ellipse))

    return arc_pairs

@jit(nopython=True)
def arc_length(arc):
    length = 0
    for line in arc:
        length += magn(line[1] - line[0])
    return length

@jit(nopython=True)
def arc_merging_degree(arc_i, arc_j, degree_i, degree_j, ellipse):
    arc_score_i, arc_score_j = 1, 1
    merging_degree = min(arc_score_i, arc_score_j)
    arc_length_i, arc_length_i = arc_length(arc_i), arc_length(arc_j)
    merging_degree *= min(arc_length_i, arc_length_i) / ellipse.perimeter
    merging_degree *= degree_i + degree_j
    return merging_degree


def merge_arcs(arcs, unused_line_segments):
    arc_pairs = form_arc_pairs(arcs)

    for arc_pair_idx, (arc_pair, ellipse) in enumerate(arc_pairs):
        arc_i, degree_i, range_a = arcs[arc_pair[0]]
        arc_j, degree_j, range_b = arcs[arc_pair[1]]
        merging_degree = arc_merging_degree(arc_i, arc_j, degree_i, degree_j, ellipse)
        arc_pairs[arc_pair_idx] = (arc_pair, ellipse, merging_degree)

    candidate_ellipse_sets = []
    for arc_idx in range(len(arcs)):
        candidate_ellipse_sets.append([arc_idx])

    # Sort by merging degree
    arc_pairs.sort(key = lambda x: x[2], reverse = True)

    for arc_pair_idx, (arc_pair, ellipse, merging_degree) in enumerate(arc_pairs):
        if merging_degree < 0.1:
            continue

        arc_i_idx, arc_j_idx = arc_pair

        # Find corresponding candidate ellipse sets
        ce_i_idx, ce_j_idx = -1, -1
        for ce_idx, ce_set in enumerate(candidate_ellipse_sets):
            if arc_i_idx in ce_set:
                ce_i_idx = ce_idx
            if arc_j_idx in ce_set:
                ce_j_idx = ce_idx
        ces_i = candidate_ellipse_sets[ce_i_idx]
        ces_j = candidate_ellipse_sets[ce_j_idx]

        # Check for conflicting pairs
        has_conflict = False
        for ai in ces_i:
            for aj in ces_j:
                has_pair = False
                for arc_pair in arc_pairs:
                    if ai in arc_pair[0] and aj in arc_pair[0]:
                        has_pair = True
                        break
                if not has_pair:
                    has_conflict = True
                    break

        if has_conflict:
            continue

        candidate_ellipse_sets[ce_i_idx] += candidate_ellipse_sets[ce_j_idx]
        del candidate_ellipse_sets[ce_j_idx]

    # Fit ellipses
    ellipses = []
    used_lines = []
    for candidate_ellipse_set in candidate_ellipse_sets:
        arcs_in_set = []
        for arc_idx in candidate_ellipse_set:
            arcs_in_set.append(arcs[arc_idx][0])
        fitting_contition, ellipse = test_fitting_condition(arcs_in_set, None)
        if fitting_contition and ellipse is not None:
            ellipses.append(ellipse)
            for arc in arcs_in_set:
                for line in arc:
                    used_lines.append(line)

    # Unused line segments (for finding the bottom parts of the goals later)
    for arc in arcs:
        for line in arc[0]:
            found = False
            for used_line in used_lines:
                if np.array_equal(used_line, line) or np.array_equal(used_line, line[::-1]):
                    found = True
                    break
            if not found:
                unused_line_segments.append(line)

    return ellipses

#@jit(nopython=True)
def form_goal_line_pairs(line_segments):
    d_max = 5.0
    line_pairs = np.zeros((100, 2, 2, 2))
    merging_degrees = np.zeros((100,))
    pair_count = 0

    for i in range(line_segments.shape[0]):
        for j in range(i + 1, line_segments.shape[0]):
            line_pairs[pair_count][0] = line_segments[i]
            line_pairs[pair_count][1] = line_segments[j]
            d = min_distance(line_pairs[pair_count][0], line_pairs[pair_count][1])
            if d < d_max:
                line_pairs[pair_count][0][:] = line_pairs[pair_count][0][::-1]
                merging_degrees[pair_count] = 1 / (d + 1)
                pair_count += 1
                if pair_count == merging_degrees.shape[0]:
                    return line_pairs, merging_degrees

    return line_pairs[0:pair_count], merging_degrees[0:pair_count]

def merge_goal_line_pairs(polytrains, line_pairs, merging_degrees):
    for line_pair, merging_degree in zip(line_pairs, merging_degrees):
        line_i, line_j = line_pair

        # Check if lines are in any polytrain
        pt_idx_line_i, pt_idx_line_j = -1, -1
        for pt_idx, pt in enumerate(polytrains):
            if np.array_equal(pt[0], line_j):
                pt_idx_line_j = pt_idx
            if np.array_equal(pt[-1], line_i):
                pt_idx_line_i = pt_idx

        if pt_idx_line_i >= 0 and pt_idx_line_j >= 0:
            pt_i = polytrains[pt_idx_line_i]
            pt_j = polytrains[pt_idx_line_j]

            polytrains[pt_idx_line_i] = pt_i + pt_j
            del polytrains[pt_idx_line_j]

        elif pt_idx_line_i >= 0:
            pt_i = polytrains[pt_idx_line_i]
            pt_i.append(line_j)

        elif pt_idx_line_j >= 0:
            pt_j = polytrains[pt_idx_line_j]
            pt_j.insert(0, line_i)

        else:
            polytrain = [line_i, line_j]
            polytrains.append(polytrain)

    # Flatten polytrain
    for idx, pt in enumerate(polytrains):
        pt_flat = [pt[0][0]]
        for line in pt:
            if not np.array_equal(pt_flat[-1], line[0]):
                print("not equal", pt_flat[-1], line[0])
                pt_flat.append(line[0])
            pt_flat.append(line[1])
        polytrains[idx] = np.array(pt_flat)


def reverse_polytrain(polytrain):



    print("reverse", polytrain)
    quit()

def merge_goal_line_segments(ellipse_outer, ellipse_inner, line_segments):
    # Form the set LPij and sort them according to merging degree
    line_pairs, merging_degrees = form_goal_line_pairs(line_segments)
    sorted_pairs = sorted(zip(line_pairs, merging_degrees), key=lambda i: i[1], reverse=True)
    line_pairs, merging_degrees = zip(*sorted_pairs)
    line_pairs, merging_degrees = list(line_pairs), list(merging_degrees)

    polytrains = []
    merge_goal_line_pairs(polytrains, line_pairs, merging_degrees)

    # Add unused line_segments to polytrains
    for line_segment in line_segments:
        used = False
        for polytrain in polytrains:
            for idx in range(1, len(polytrain)):
                if np.array_equal(line_segment, polytrain[idx-1:idx+1]) or \
                   np.array_equal(line_segment, polytrain[idx-1:idx+1][::-1]):
                    used = True
                    break
            if used:
                break
        if not used:
            polytrains.append(line_segment)

    # Compute fitness
    fitnesses = [] 
    for idx, pt in enumerate(polytrains):
        end_a, end_b = pt[0], pt[-1]
        distances = (ellipse_outer.point_distance(end_a),
                     ellipse_inner.point_distance(end_b),
                     ellipse_outer.point_distance(end_b),
                     ellipse_inner.point_distance(end_a))
        fitness_a = 10 / (distances[0]**2 + distances[1]**2)
        fitness_b = 10 / (distances[2]**2 + distances[3]**2)
        if fitness_b > fitness_a:
            # Polytrain should go from outer to inner ellipse
            polytrains[idx] = pt[::-1]
        fitnesses.append(max(fitness_a, fitness_b))

    # Remove bad polytrains
    fitnesses, polytrains = zip(*sorted(zip(fitnesses, polytrains), reverse=True))
    fitness_threshold = 0.01
    cutoff_idx = len(polytrains)
    for idx in range(len(polytrains)):
        if fitnesses[idx] < fitness_threshold:
            cutoff_idx = idx
            break
    fitnesses, polytrains = fitnesses[0:cutoff_idx], polytrains[0:cutoff_idx]

    # Find most suited pair
    best_idx_i, best_idx_j, best_fitness = -1, -1, -1
    for idx_i in range(len(polytrains)):
        for idx_j in range(idx_i + 1, len(polytrains)):
            p_i, p_j = polytrains[idx_i][0], polytrains[idx_j][0]
            v_i = p_i - ellipse_outer.center
            v_j = p_j - ellipse_outer.center
            angle = vector_intersection_angle(v_i, v_j)
            fitness = fitnesses[idx_i] * fitnesses[idx_j] / (abs(angle - 2.7) + 1)
            if fitness > best_fitness:
                best_fitness = fitness
                best_idx_i, best_idx_j = idx_i, idx_j

    if best_fitness < 0:
        # Failed to find goal ends
        return None

    goal_end_points = np.concatenate((polytrains[best_idx_i], polytrains[best_idx_j]))
    return goal_end_points


def merge_ellipses(ellipses, line_segments):

    # Sort by area
    ellipses.sort(key = lambda x: x.area, reverse = True)

    ellipse_pairs = []
    for idx_i in range(len(ellipses)):
        for idx_j in range(idx_i + 1, len(ellipses)):
            ellipse_i, ellipse_j = ellipses[idx_i], ellipses[idx_j]
            area_ratio = ellipse_j.area / ellipse_i.area
            distance = magn(ellipse_j.center - ellipse_i.center)
            axis = min(ellipse_i.axes)
            max_distance = axis * 0.25

            # Improvement: Prevent conflicting pairs

            if area_ratio < 1.0 and area_ratio > 0.5 and distance < max_distance:
                #print("===")
                #print(area_i, area_j)
                #print(center_i, center_j)
                #print("d", distance)
                #print("a", axis)
                ellipse_pairs.append((idx_i, idx_j))
                break

    line_segments = np.array(line_segments, dtype=np.float64)
    goals = []

    for ellipse_pair in ellipse_pairs:
        ellipse_i = ellipses[ellipse_pair[0]]
        ellipse_j = ellipses[ellipse_pair[1]]
        lines = []

        for line in line_segments:
            if ellipse_i.is_inside(line[0], 1.03) and \
               ellipse_i.is_inside(line[1], 1.03) and \
               not ellipse_j.is_inside(line[0], 0.97) and \
               not ellipse_j.is_inside(line[1], 0.97):
                lines.append(line)

        lines = np.array(lines, dtype=np.float64)

        goal_end_points = merge_goal_line_segments(ellipse_i, ellipse_j, lines)
        if goal_end_points is not None:
            goals.append((ellipse_i, ellipse_j, goal_end_points))


    import random
    from seaborn import color_palette
    palette = color_palette("husl", len(goals))
    image = skimage.io.imread("img_in.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int32)
    #im2 = image

    for idx, (ellipse_outer, ellipse_inner, goal_end_points) in enumerate(goals):
        color = (palette[idx][0] * 255, palette[idx][1] * 255, palette[idx][2] * 255)

        for ellipse in (ellipse_inner, ellipse_outer):
            cv2.ellipse(im2, box = ellipse.to_tuple(), color = color)


        def ellipse_line_intersection(points):

            ellipse = ellipse_outer

            ps = np.empty_like(points)
            for idx, point in enumerate(points):
                ps[idx] = rotate_vector_2d(point - ellipse.center, -ellipse.angle)

            slope, intercept, r_value, p_value, std_err = stats.linregress(ps)
            print(slope, intercept, r_value, p_value, std_err)


            a = ellipse.axes[1]**2 + ellipse.axes[0]**2 * slope**2
            b = 2 * slope * ellipse.axes[0]**2 * intercept
            c = ellipse.axes[0]**2 * intercept**2 - ellipse.axes[0]**2 * ellipse.axes[1]**2

            print("a, b, c", a, b, c)

            s = b**2 - 4 * a * c
            if s < 0:
                print("no hit!")
                quit()

            s0 = 1 / (2 * a) * (-b + math.sqrt(s))
            s1 = 1 / (2 * a) * (-b - math.sqrt(s))

            p = np.array(((s0, slope * s0 + intercept),
                          (s1, slope * s1 + intercept)))

            p[0] = rotate_vector_2d(p[0], ellipse.angle) + ellipse.center
            p[1] = rotate_vector_2d(p[1], ellipse.angle) + ellipse.center

            print("s0, s1", s0, s1)
            print("p", p)


            print("line_intersection")
            print("slope, intercept", slope, intercept)

            return p

        #p0, p1 = ellipse.line_intersection(goal_end)


        p0 = ellipse_line_intersection(goal_end_points)

        def ln(lln, col):
            rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
            im2[rr, cc] = col

        try:
            ln(p0, color)
        except:
            pass

        #for idx in range(1, len(goal_end_points)):
        #    ln(goal_end_points[idx-1:idx+1], color)


        #def ln(lln, col):
        #    rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
        #    im2[rr, cc] = col

        #ln(line, color)

        
    skimage.io.imsave("img_ellipses.png", (im2).astype(np.uint8))
    #input("Press Enter to continue...")


    print("end")


    quit()


import pickle
"""
if False:
    image = read_image("img_in.png")
    strips = extract_strips(image)
    line_segments = split_strips(strips)

    with open('line_segments.dat', 'wb') as f:
        pickle.dump(line_segments, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('line_segments.dat', 'rb') as f:
        line_segments = pickle.load(f)

arcs, unused_line_segments, merge_line_segments(line_segments)

with open('arcs.dat', 'wb') as f:
    pickle.dump((arcs, unused_line_segments), f, pickle.HIGHEST_PROTOCOL)
quit()
"""
with open('arcs.dat', 'rb') as f:
    arcs, unused_line_segments = pickle.load(f)

ellipses = merge_arcs(arcs, unused_line_segments)

goals = merge_ellipses(ellipses, unused_line_segments)


"""
    import random
    from seaborn import color_palette
    palette = color_palette("husl", len(unused_line_segments))
    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3))

    for idx, line in enumerate(unused_line_segments):
        cl = palette[idx]

        def ln(lln, col):
            rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
            im2[rr, cc] = col

        ln(line, cl)

    skimage.io.imsave("img_unused_1.png", (im2 * 255).astype(np.uint8))


    unused_line_segments.clear()
"""

