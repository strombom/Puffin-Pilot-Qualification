
import copy
import math
import skimage
import skimage.morphology
import numpy as np
import cv2
from numba import jit, njit


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
def intersection_angle(l1, l2):
    v1 = l1[1] - l1[0]
    v2 = l2[1] - l2[0]
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
            theta = abs(intersection_angle(line_pairs[pair_count][0], line_pairs[pair_count][1]))
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
    theta = abs(intersection_angle((aj, bj), (ak, bk)))
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

@jit(nopython=True)
def ellipse_point_distance(e_center, e_axes, e_angle, p):
    # https://github.com/0xfaded/ellipse_demo/issues/1
    p = np.abs(rotate_vector_2d(p - e_center, -e_angle))
    tx, ty = 0.707106, 0.707106
    a, b = e_axes / 2
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

@jit(nopython=True)
def ellipse_fitting_score(e_center, e_axis, e_angle, points):
    score = 0
    for idx in range(points.shape[0]):
        d = ellipse_point_distance(e_center, e_axis, e_angle, points[idx])
        if d <  ellipse_fitting_threshold:
            score += 1
    return score / points.shape[0]

def test_fitting_condition(arcs, lines = None):
    arc_points = []
    for arc in arcs:
        for line in arc:
            arc_points.append(line[0])
            arc_points.append(line[1])
    arc_points = np.unique(arc_points, axis=0)

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
    ellipse = cv2.fitEllipseDirect(points)

    e_center = np.array(ellipse[0])
    e_axes = np.array(ellipse[1])
    e_angle = np.radians(ellipse[2])

    score_arc = ellipse_fitting_score(e_center, e_axes, e_angle, arc_points)
    if len(line_points) > 0:
        score_line = ellipse_fitting_score(e_center, e_axes, e_angle, line_points)
    else:
        score_line = 1.0

    threshold_fit = 0.8

    if score_arc < threshold_fit or score_line < threshold_fit:
        return False, ellipse

    return True, ellipse

def merge_line_pairs(arcs, line_pairs, merging_degrees, first = True):

    for line_pair, merging_degree in zip(line_pairs, merging_degrees):
        line_i, line_j = line_pair

        # Check if lines are in any arc
        arc_idx_line_i, arc_idx_line_j = -1, -1
        for arc_idx, arc in enumerate(arcs):
            if np.array_equal(arc[0], line_j):
                arc_idx_line_j = arc_idx
            if np.array_equal(arc[-1], line_i):
                arc_idx_line_i = arc_idx

        mi, mj = [], []
        for arc_idx, arc in enumerate(arcs):
            if np.array_equal(arc[0], line_j):
                mj.append(arc_idx)
            if np.array_equal(arc[-1], line_i):
                mi.append(arc_idx)

        if len(mi) > 1 or len(mj) > 1:
            print(mi, mj)
            quit()

        if arc_idx_line_i >= 0 and arc_idx_line_j >= 0:
            arc_i = arcs[arc_idx_line_i]
            arc_j = arcs[arc_idx_line_j]

            if test_line_merging_condition(line_i, line_j):
                if test_fitting_condition([arc_i, arc_j], [])[0]:
                    arcs[arc_idx_line_i] = arc_i + arc_j
                    del arcs[arc_idx_line_j]
                    merging_lines = True

        elif arc_idx_line_i >= 0:
            # Join line j into arc
            arc_i = arcs[arc_idx_line_i]

            if test_line_merging_condition(arc_i[-2], line_j):
                if test_fitting_condition([arc_i], [line_j])[0]:
                    arc_i.append(line_j)
            #else:
            #    print("Angle and length condition failed")
            #    quit()

        elif arc_idx_line_j >= 0:
            # Join line i into arc
            arc_j = arcs[arc_idx_line_j]

            if test_line_merging_condition(arc_j[0], line_i):
                if test_fitting_condition([arc_j], [line_j])[0]:
                    arc_j.insert(0, line_i)
            #else:
            #    print("Angle and length condition failed")
            #    quit()

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
    merge_line_pairs(arcs, line_pairs_bad, merging_degrees_bad, first = False)
    
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
            a = intersection_angle(arc_i[-1], line_1)
            b = intersection_angle(line_1, arc_j[0])
            c = intersection_angle(arc_j[-1], line_2)
            d = intersection_angle(line_2, arc_i[0])
            if a < 0 or b < 0 or c < 0 or d < 0:
                # At least one intersection has the wrong direction
                continue
            
            # Check fitting condition
            fitting_contition, ellipse = test_fitting_condition([arc_i, arc_j], None)
            if fitting_contition:
                arc_pairs.append(((arc_idx_i, arc_idx_j), ellipse))

    return arc_pairs

@jit(nopython=True)
def ellipse_perimeter(ellipse):
    a, b = ellipse[1][0], ellipse[1][1]
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

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
    merging_degree *= min(arc_length_i, arc_length_i) / ellipse_perimeter(ellipse)
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

def merge_ellipses(ellipses):

    # Sort by area
    for idx, ellipse in enumerate(ellipses):
        area = math.pi * ellipse[1][0] / 2 * ellipse[1][1] / 2
        ellipses[idx] = (ellipse, area)
    ellipses.sort(key = lambda x: x[1], reverse = True)

    ellipse_pairs = []
    for idx_i in range(len(ellipses)):
        for idx_j in range(idx_i + 1, len(ellipses)):
            area_i, area_j = ellipses[idx_i][1], ellipses[idx_j][1]
            area_ratio = area_j / area_i
            center_i = np.array(ellipses[idx_i][0][0])
            center_j = np.array(ellipses[idx_j][0][0])
            distance = magn(center_j - center_i)
            axis = min(ellipses[idx_i][0][1])
            max_distance = axis * 0.25

            if area_ratio < 1.0 and area_ratio > 0.5 and distance < max_distance:
                print("===")
                print(area_i, area_j)
                print(center_i, center_j)
                print("d", distance)
                print("a", axis)
                ellipse_pairs.append((idx_i, idx_j))
                break



    import random
    from seaborn import color_palette
    palette = color_palette("husl", len(ellipse_pairs))
    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.int32)


    for ep_idx, (e_idx_i, e_idx_j) in enumerate(ellipse_pairs):

        color = (palette[ep_idx][0] * 255, palette[ep_idx][1] * 255, palette[ep_idx][2] * 255)

        cv2.ellipse(im2, box = ellipses[e_idx_i][0], color = color)
        cv2.ellipse(im2, box = ellipses[e_idx_j][0], color = color)

        """
        center = np.array(ellipse[0])
        axis = np.array((0, -ellipse[1][1])) / 2
        axis = rotate_vector_2d(axis, np.radians(ellipse[2]))
        endpoint = center + axis


        def ln(lln, col):
            rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
            im2[rr, cc] = col

        try:
            ln(np.array((center, endpoint)), color)
        except:
            pass
        
        print(center, endpoint)
        """

        #print(center)
        #quit()
        
        skimage.io.imsave("img_line_pairs.png", (im2).astype(np.uint8))
        input("Press Enter to continue...")





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

