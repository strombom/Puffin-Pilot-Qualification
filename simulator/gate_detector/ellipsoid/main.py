
import copy
import skimage
import skimage.morphology
import numpy as np
from numba import jit, njit

image = skimage.io.imread("img_in.png")

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


def split_strips(strips, line_segments):
    segment_count = 0
    minimum_segment_length = 6
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
        l_dir = np.zeros(2)

        for idx in range(1, strip_length):
            if segment_count == line_segments.shape[0]:
                return segment_count

            segment_length = idx - line_start_idx

            # Last point
            if idx == strip_length - 1:
                if segment_length > minimum_segment_length:
                    print("lastpoint")
                    line_segments[segment_count] = strip[line_start_idx], strip[idx]
                    segment_count += 1
                break

            # Calculate line direction
            p_dir = strip[idx] - strip[line_start_idx]
            p_dir = p_dir / np.linalg.norm(p_dir)
            l_dir = (l_dir * segment_length + p_dir) / (segment_length + 1)
            l_dir = l_dir / np.linalg.norm(l_dir)

            # Check if current point is outside of the lien
            distance = np.linalg.norm(p_dir - l_dir) * segment_length
            if distance > maximum_point_to_line_distance:
                # Store line segment
                line_segments[segment_count] = strip[line_start_idx], strip[idx]
                segment_count += 1

                # Initialize next line segment
                line_start_idx = idx
                l_dir[:] = (0, 0)

    return segment_count






strips = extract_strips(image)

line_segments = np.zeros((1000,2,2), dtype=np.int)
line_segment_count = split_strips(strips, line_segments)



import random
import seaborn as sns
palette = sns.color_palette("hls", line_segment_count)

im2 = np.ones((image.shape[0], image.shape[1], 3))
for idx in range(line_segment_count):
    p0, p1 = line_segments[idx]

    #print(p0, p1)

    rr, cc = skimage.draw.line(p0[1], p0[0], p1[1], p1[0])
    im2[rr, cc] = palette[random.randrange(line_segment_count)]

skimage.io.imsave("img_line_segments.png", im2)
quit()




print(strips)
quit()

skimage.io.imshow(image)
skimage.io.show()


