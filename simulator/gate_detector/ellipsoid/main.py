
import copy
import math
import skimage
import skimage.morphology
import numpy as np
import cv2
from numba import jit, njit


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

    """
    import seaborn as sns
    palette = sns.color_palette("hls", len(strips)) # strip_idx)

    im2 = np.ones((image.shape[0], image.shape[1], 3)) #, dtype=np.int8)
    for idx, strip in enumerate(strips):
        for point in strip:
            if point[0] == 0 and point[1] == 0:
                break
            im2[point[1], point[0]] = palette[idx]

    skimage.io.imsave("img_extr.png", im2)
    quit()
    """

    return strips


@jit(nopython=True)
def magn(v):
    return (v[0] * v[0] + v[1] * v[1]) ** 0.5

@jit(nopython=True)
def split_strips(strips):
    line_segments = np.zeros((1000,2,2), dtype=np.int32)

    segment_count = 0
    minimum_segment_length = 10
    maximum_point_to_line_distance = 2.0

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
    # Find the two closest points in the line segments
    dmin = 99999
    min_1, min_2 = 0, 0
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
    return abs(math.atan2(det, dot))

@jit(nopython=True)
def merging_degree(d, theta, l1, l2):
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
    theta_max = 189.9 / 180 * math.pi
    line_pairs = np.zeros((1000, 2, 2, 2))
    merging_degrees = np.zeros((1000,))
    pair_count = 0

    for i in range(line_segments.shape[0]):
        for j in range(i + 1, line_segments.shape[0]):
            d = min_distance(line_segments[i], line_segments[j])
            theta = intersection_angle(line_segments[i], line_segments[j])
            if d < d_max and theta > theta_min and theta < theta_max:
                line_pairs[pair_count][0] = line_segments[i]
                line_pairs[pair_count][1] = line_segments[j]
                make_line_pair_clockwise(line_pairs[pair_count])
                li, lj = line_length(line_segments[i]), line_length(line_segments[j])
                merging_degrees[pair_count] = merging_degree(d, theta, li, lj)
                pair_count += 1

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
    theta = intersection_angle((aj, bj), (ak, bk))
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

#@jit(nopython=True)
def ellipse_fitting_score(e_center, e_axis, e_angle, points):
    score = 0
    for idx in range(points.shape[0]):
        d = ellipse_point_distance(e_center, e_axis, e_angle, points[idx])
        if d <  1.5:
            score += 1
        else:
            print(d)
    return score / points.shape[0]


plotcon = 0

def test_fitting_condition(arcs, lines):
    arc_points = []
    for arc in arcs:
        for line in arc:
            arc_points.append(line[0])
            arc_points.append(line[1])
    arc_points = np.unique(arc_points, axis=0)

    line_points = []
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
        return True

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
    global plotcon
    plotcon += 1
    if plotcon == 3000 or score_arc < threshold_fit or score_line < threshold_fit:
        print("test_fitting_condition false", score_arc, score_line)
        print(arc_points)
        print(line_points)
        """
        import seaborn as sns
        image = skimage.io.imread("img_line_segments.png")
        im2 = np.zeros((image.shape[0], image.shape[1], 3))
        im2 = (im2 * 255).astype(np.int32)
        cv2.ellipse(im2, ellipse, (0,255,0),1)
        def pp(pnt, col = (1, 0, 0)):
            im2[int(pnt[1]), int(pnt[0])] = col
        for point in points:
            pp(point, (255,0,0))
        skimage.io.imsave("img_line_pairs.png", (im2).astype(np.uint8))
        quit()
        """
        return False

    return True

    """
    import seaborn as sns
    #palette = sns.color_palette("hls", 2)
    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3))
    #def ln(lln, col):
    #    rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
    #    im2[rr, cc] = col
    #ln(arc[-2], (0, 0, 0.3))
    #ln(line_j, (0, 0.3, 0))
    im2 = im2 * 255
    im2 = im2.astype(np.int32)
    cv2.ellipse(im2, ellipse, (0,255,0),1)
    def pp(pnt, col = (1, 0, 0)):
        im2[int(pnt[1]), int(pnt[0])] = col
    for point in points:
        pp(point, (255,0,0))
    #center = (int(ellipse[0][0]), int(ellipse[0][1]))
    #axes = (int(ellipse[1][0]), int(ellipse[1][1]))
    #angle = int(ellipse[2])
    skimage.io.imsave("img_line_pairs.png", (im2).astype(np.uint8))
    quit()
    """

def merge_line_pairs(arcs, line_pairs, merging_degrees):
    first = True
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
            #line_i, line_j = arc_i[-2], arc_j[0]
            if test_line_merging_condition(line_i, line_j):
                if test_fitting_condition([arc_i, arc_j], []):
                    arcs[arc_idx_line_i] = arc_i + arc_j
                    del arcs[arc_idx_line_j]

        elif arc_idx_line_i >= 0:
            # Join line j into arc
            #print("join j arc")
            arc = arcs[arc_idx_line_i]
            #line_i, line_j = arc_i[-2], arc_j[0]

            if test_line_merging_condition(arc[-2], line_j):
                if test_fitting_condition([arc], [line_j]):
                    arc.append(line_j)
            else:
                print("Angle and length condition failed")
                quit()

        elif arc_idx_line_j >= 0:
            # Join line i into arc
            #print("join i arc")
            arc = arcs[arc_idx_line_j]

            if test_line_merging_condition(arc[0], line_i):
                if test_fitting_condition([arc], [line_j]):
                    arc.insert(0, line_i)
            else:
                print("Angle and length condition failed")
                quit()

        else:
            # Create new arc
            arc = [line_i, line_j]
            arcs.append(arc)


    # Join arcs

    print("end")
    print(len(arcs))
    import random
    import seaborn as sns
    palette = sns.color_palette("hls", 20)
    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3))
    #im2 = (im2).astype(np.int32)
    for arc in arcs:

        cl = palette[random.randrange(20)]
        print(" ", len(arc))
        #cv2.ellipse(im2, ellipse, (0,255,0),1)


        def ln(lln, col):
            rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
            im2[rr, cc] = col #(1, 0.5, 0) #col
            #print(col)

        for line in arc:
            ln(line, cl)

        #def pp(pnt, col = (1, 0, 0)):
        #    im2[int(pnt[1]), int(pnt[0])] = col
        #for point in points:
        #    pp(point, (255,0,0))

    im2 *= 255
    skimage.io.imsave("img_line_pairs.png", (im2).astype(np.uint8))
    quit()




"""
if length_condition > 0.8:
    # Length condition not met
    print("arc\n", arc)
    print("aj", aj)
    print("bj", bj)
    print("ak", ak)
    print("bk", bk)
    print("m", m)
    print("length_condition", length_condition)

    import seaborn as sns
    #palette = sns.color_palette("hls", 2)

    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3))

    def ln(lln, col):
        rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
        im2[rr, cc] = col

    ln(arc[-2], (0, 0, 0.3))
    ln(line_j, (0, 0.3, 0))

    def pp(pnt, col = (1, 0, 0)):
        im2[int(pnt[1]), int(pnt[0])] = col

    pp(m, (1,0,0))
    pp(aj, (0,1,0))
    pp(bj, (0,1,0))
    pp(ak, (0,0,1))
    pp(bk, (0,0, 1))

    skimage.io.imsave("img_line_pairs.png", (im2 * 255).astype(np.uint8))
    quit()

"""

"""
    import seaborn as sns
    #palette = sns.color_palette("hls", 2)

    image = skimage.io.imread("img_line_segments.png")
    im2 = np.zeros((image.shape[0], image.shape[1], 3))
    for idx in range(len(line_pairs)):
        if idx % 3 == 1:
            p0, p1 = line_pairs[idx][0]

            rr, cc = skimage.draw.line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            im2[rr, cc, 0] = 1
            #im2[rr, cc, 1] = 0
            #im2[rr, cc, 2] = 0

            p0, p1 = line_pairs[idx][1]
            rr, cc = skimage.draw.line(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            #im2[rr, cc, 0] = 0
            im2[rr, cc, 1] = 1
            #im2[rr, cc, 2] = 0
            #break

    skimage.io.imsave("img_line_pairs.png", (im2 * 255).astype(np.uint8))
    quit()

    print(line_pairs)
    quit()
"""


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


    return

import pickle

if False:
    image = read_image("img_in.png")
    strips = extract_strips(image)
    line_segments = split_strips(strips)

    with open('line_segments.dat', 'wb') as f:
        pickle.dump(line_segments, f, pickle.HIGHEST_PROTOCOL)
else:
    with open('line_segments.dat', 'rb') as f:
        line_segments = pickle.load(f)



arcs = merge_line_segments(line_segments)


import random
import seaborn as sns
palette = sns.color_palette("hls", line_segments.shape[0])

im2 = np.ones((image.shape[0], image.shape[1], 3))
for idx in range(line_segments.shape[0]):
    p0, p1 = line_segments[idx]

    #print(p0, p1)

    rr, cc = skimage.draw.line(p0[1], p0[0], p1[1], p1[0])
    im2[rr, cc] = palette[random.randrange(line_segments.shape[0])]


skimage.io.imsave("img_line_segments.png", (im2 * 255).astype(np.uint8))
quit()




print(strips)
quit()

skimage.io.imshow(image)
skimage.io.show()


