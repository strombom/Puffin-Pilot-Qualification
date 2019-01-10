
import copy
import math
import skimage
import skimage.morphology
import numpy as np
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

            # Check if current point is outside of the lien
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
def form_line_pairs(line_segments):
    d_max = 5.0
    theta_min = 145.0 / 180 * 3.14
    line_pairs = np.zeros((1000, 2, 2, 2))
    merging_degrees = np.zeros((1000,))
    pair_count = 0

    for i in range(line_segments.shape[0]):
        for j in range(i + 1, line_segments.shape[0]):
            d = min_distance(line_segments[i], line_segments[j])
            theta = intersection_angle(line_segments[i], line_segments[j])
            if d < d_max and theta > theta_min:
                line_pairs[pair_count][0] = line_segments[i]
                line_pairs[pair_count][1] = line_segments[j]
                li, lj = line_length(line_segments[i]), line_length(line_segments[j])
                merging_degrees[pair_count] = merging_degree(d, theta, li, lj)
                pair_count += 1

    return line_pairs[0:pair_count], merging_degrees[0:pair_count]


def merge_line_segments(line_segments):

    # Form the set LPij and sort them according to merging degree
    line_pairs, merging_degrees = form_line_pairs(line_segments)
    sorted_pairs = sorted(zip(line_pairs, merging_degrees), key=lambda i: i[1], reverse=True)
    line_pairs, merging_degrees = zip(*sorted_pairs)

    print(line_pairs[0:2])
    print(merging_degrees[0:2])
    quit()

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


