
import copy
import math
import skimage
import skimage.morphology
import numpy as np
import cv2
from scipy import stats
from numba import jit, njit, jitclass, int32, float32, float64




def read_image(filename):
    image = skimage.io.imread(filename)

    # Skeletonize image
    #thresh = skimage.filters.threshold_minimum(image, nbins=256, max_iter=10000)
    thresh = 110
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
def line_length(line_segment):
    return magn(line_segment[0] - line_segment[1])

@jit(nopython=True)
def split_strips(strips):
    line_segments = np.zeros((1000,2,2)) #, dtype=np.int32)

    segment_count = 0
    minimum_segment_length = 7
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
                line_segments[segment_count][1] = strip[idx - 1]
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
def line_merging_degree(d, theta):
    return 1.0 / (d*0.2 + 1) * math.cos((math.pi - theta)**0.2)

@jit(nopython=True)
def form_line_pairs(line_segments):
    d_max = 3.0
    theta_min = 160.0 / 180 * math.pi
    theta_max = 10.0 / 180 * math.pi
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
            if d < d_max and theta > theta_min:
                line_pairs[pair_count][0][:] = line_pairs[pair_count][0][::-1]
                merging_degrees[pair_count] = line_merging_degree(d, theta)
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
def rotate_vector_2d(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    r = np.array(((c, -s), (s, c)))
    return np.dot(r, v)

def test_fitting_condition(start_point, end_point, line_trains, lines = None, debug = False):
    lt_points = []
    for line_train in line_trains:
        for line in line_train:
            lt_points.append(line[0])
            lt_points.append(line[1])
    lt_points = np.unique(lt_points, axis=0).astype(np.float64)
    line_points = []
    if lines is not None:
        for line in lines:
            line_points.append(line[0])
            line_points.append(line[1])
    if len(line_points) > 0:
        line_points = np.unique(line_points, axis=0)
        points = np.vstack((lt_points, line_points))
        points = np.unique(points, axis=0)
    else:
        points = lt_points

    if len(points) < 3:
        return False

    line = (start_point, end_point)
    x1, y1 = line[0]
    x2, y2 = line[1]
    div = np.sqrt(np.square(x2-x1) + np.square(y2-y1))
    for point in points:
        distance = abs((x2-x1)*(y1-point[1]) - (x1-point[0])*(y2-y1)) / div
        if distance > 10.0:
            return False
    return True

def merge_line_pairs(line_trains, line_pairs):
    for line_pair in line_pairs:
        line_i, line_j = line_pair

        # Check if lines are in any line_train
        lt_idx_line_i, lt_idx_line_j = -1, -1
        for lt_idx, line_train in enumerate(line_trains):
            if np.array_equal(line_train[0], line_j):
                lt_idx_line_j = lt_idx
            elif np.array_equal(line_train[-1], line_i):
                lt_idx_line_i = lt_idx
            elif np.array_equal(line_train[0], line_j[::-1]):
                print("1")
                quit()
            elif np.array_equal(line_train[-1], line_i[::-1]):
                print("2")
                quit()
            elif np.array_equal(line_train[0], line_i):
                print("3")
                quit()
            elif np.array_equal(line_trains[-1], line_j):
                print("4")
                quit()
            elif np.array_equal(line_train[0], line_i[::-1]):
                print("5")
                quit()
            elif np.array_equal(line_trains[-1], line_j[::-1]):
                print("6")
                quit()

        if lt_idx_line_i >= 0 and lt_idx_line_j >= 0:
            lt_i = line_trains[lt_idx_line_i]
            lt_j = line_trains[lt_idx_line_j]
            if test_fitting_condition(start_point = lt_i[0][0],
                                      end_point = lt_j[-1][1],
                                      line_trains = [lt_i, lt_j],
                                      lines = []):
                line_trains[lt_idx_line_i] = lt_i + lt_j
                print("long one")
                print(lt_i + lt_j)
                quit()
                del line_trains[lt_idx_line_j]

        elif lt_idx_line_i >= 0:
            # Join line j into line_train
            lt_i = line_trains[lt_idx_line_i]
            if test_fitting_condition(start_point = lt_i[0][0],
                                      end_point = line_j[1],
                                      line_trains = [lt_i],
                                      lines = [line_j]):
                lt_i.append(line_j)

        elif lt_idx_line_j >= 0:
            # Join line i into line_train
            lt_j = line_trains[lt_idx_line_j]
            if test_fitting_condition(start_point = lt_j[-1][1],
                                      end_point = line_i[1],
                                      line_trains = [lt_j],
                                      lines = [line_i]):
                lt_j.insert(0, line_i)

        else:
            # Create new line_train
            line_train = [line_i, line_j]
            line_trains.append(line_train)


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

    line_trains = []
    merge_line_pairs(line_trains, line_pairs_good)
    merge_line_pairs(line_trains, line_pairs_bad)

    unused_line_segments = []
    for line_segment in line_segments:
        found = False
        for line_train in line_trains:
            for line in line_train:
                if np.array_equal(line_segment, line) or np.array_equal(line_segment, line[::-1]):
                    found = True
                    break
            if found:
                break
        if not found:
            if line_length(line_segment) > 12:
                unused_line_segments.append(line_segment)

    for line_segment in unused_line_segments:
        line_train.append([line_segment])

    merge i, j from line_trains, not from unused_line_segments:::\/

    merging = True
    while merging:
        merging = False
        for uls_idx, line_segment in enumerate(unused_line_segments):
            line_segment_length = line_length(line_segment)
            for lt_idx, line_train in enumerate(line_trains):
                line_p1, line_p2 = line_segment
                train_p1, train_p2 = line_train[0][0], line_train[-1][1]
                # Check if line is closest to the start or end of the line train
                # No need to check both ends of the line segment since we will
                #  make sure that line and train don't overlap later.
                train_front = True
                if magn(train_p1 - line_p1) >= magn(train_p2 - line_p1):
                    train_front = False
                if train_front:
                    if magn(train_p1 - line_p1) < magn(train_p1 - line_p2):
                        line_segment = line_segment[::-1]
                else:
                    if magn(train_p2 - line_p2) < magn(train_p2 - line_p1):
                        line_segment = line_segment[::-1]
                candidate_train = np.empty((len(line_train) + 1, 2, 2))
                if train_front:
                    candidate_train[0] = line_segment
                    candidate_train[1:len(line_train) + 1] = line_train
                else:
                    candidate_train[0:len(line_train)] = line_train
                    candidate_train[-1] = line_segment

                # Check length of candidate
                line_train_length = line_length([line_train[0][0], line_train[-1][1]])
                length_min = line_segment_length + line_train_length
                length_max = length_min * 1.2
                candidate_length = magn(candidate_train[0][0] - candidate_train[-1][1])
                if candidate_length < length_min or candidate_length > length_max:
                    continue
                fc = test_fitting_condition(start_point = candidate_train[0][0],
                                            end_point = candidate_train[-1][1],
                                            line_trains = [candidate_train])
                if fc:
                    line_trains[lt_idx] = copy.deepcopy(candidate_train)
                    del unused_line_segments[uls_idx]

                    merging = True
                    break
            if merging:
                break

    # Create all polytrains
    polytrains = []
    for line_train in line_trains:
        polytrain = []
        for ls in line_train:
            polytrain.append(ls[0])
            polytrain.append(ls[1])
        polytrains.append(np.array(polytrain))
    for ls in unused_line_segments:
        polytrains.append(ls)

    return polytrains
    """
    import random
    from seaborn import color_palette
    palette = color_palette("husl", len(polytrains))
    image = skimage.io.imread("img_in.png")
    im = np.zeros((image.shape[0], image.shape[1], 3))
    def ln(lln, col):
        rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
        im[rr, cc] = col
    #ln(line_segment, palette[0])
    for pidx, polytrain in enumerate(polytrains):
        cl = palette[pidx]
        for idx in range(1, len(polytrain)):
            p0, p1 = polytrain[idx-1], polytrain[idx]
            ln((p0, p1), cl)
    skimage.io.imsave("img_merge2.png", (im * 255).astype(np.uint8))
    quit()
    """



def form_poly_pairs(polytrains):
    polytrain_pairs = []

    # Calculate polytrain lengths
    #for pt_idx, polytrain in enumerate(polytrains):
    #    polytrains[pt_idx] = (polytrain, line_length(polytrain[[0,-1]]))

    # Evaluate arc pairs
    for pt_idx_i in range(len(polytrains)):
        for pt_idx_j in range(pt_idx_i + 1, len(polytrains)):
            pt_i = polytrains[pt_idx_i]
            pt_j = polytrains[pt_idx_j]

            line_i, line_j = pt_i[[0,-1]], pt_j[[0,-1]]
            d = [magn(line_i[0] - line_j[0]),
                 magn(line_i[0] - line_j[1]),
                 magn(line_i[1] - line_j[0]),
                 magn(line_i[1] - line_j[1])]
            i = np.argmin(d)
            distance = d[i]

            # Check that the corners are close enough
            if distance > 15:
                continue
            if i == 1 or i == 3:
                line_j[:] = line_j[::-1]
            if i == 2 or i == 3:
                line_i[:] = line_i[::-1]
            angle = abs(line_intersection_angle(line_i, line_j))
            if angle < 1.05 or angle > 2.1:
                continue

            merging_degree = 1 / (distance + 1)
            polytrain_pairs.append(((pt_idx_i, pt_idx_j), merging_degree))

    return polytrain_pairs

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


def merge_polytrains(polytrains):
    poly_pairs = form_poly_pairs(polytrains)

    import random
    from seaborn import color_palette
    palette = color_palette("husl", len(poly_pairs))
    image = skimage.io.imread("img_in.png")
    im = np.zeros((image.shape[0], image.shape[1], 3))
    def ln(lln, col):
        rr, cc = skimage.draw.line(int(lln[0][1]), int(lln[0][0]), int(lln[1][1]), int(lln[1][0]))
        im[rr, cc] = col
    #ln(line_segment, palette[0])
    for pidx, poly_pair in enumerate(poly_pairs):

        for ptr_idx in poly_pair[0]:
            polytrain = polytrains[ptr_idx]
            cl = palette[pidx]
            for idx in range(1, len(polytrain)):
                p0, p1 = polytrain[idx-1], polytrain[idx]
                ln((p0, p1), cl)
    skimage.io.imsave("img_merge3.png", (im * 255).astype(np.uint8))
    quit()


    print(poly_pairs)
    quit()

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


if __name__ == '__main__':
    import pickle

    
    if True:
        image = read_image("img_in.png")
        strips = extract_strips(image)
        line_segments = split_strips(strips)

        with open('line_segments.dat', 'wb') as f:
            pickle.dump(line_segments, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open('line_segments.dat', 'rb') as f:
            line_segments = pickle.load(f)

    print("1")
    polytrains = merge_line_segments(line_segments)
    print("2")


    #with open('polytrains.dat', 'wb') as f:
    #    pickle.dump(polytrains, f, pickle.HIGHEST_PROTOCOL)
    #quit()
    

    with open('polytrains.dat', 'rb') as f:
        polytrains = pickle.load(f)

    goals = merge_polytrains(polytrains)

    print(goals)
    quit()

    #goals = merge_ellipses(ellipses, unused_line_segments)


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

