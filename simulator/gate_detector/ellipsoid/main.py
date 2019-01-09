
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

#@jit(nopython=True)
def extract_strip(image, ox, oy, strip):
    idx = 0
    first_direction = True
    first_point = True
    restart = False
    x, y = ox, oy

    while True:
        if idx == strip.shape[0]:
            return idx

        if restart:
            x, y = ox, oy
            restart = False
        else:
            strip[idx] = (x, y)
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
            if first_direction:
                first_direction = False
                restart = True
                continue
            else:
                # End of strip
                break

        elif neighbour_count == 1:
            dx, dy = get_neighbour()
            x, y = x + dx, y + dy
            continue
        
        else:
            if first_point:
                first_point = False
                dx, dy = get_neighbour()
                x, y = x + dx, y + dy
                continue
            else:
                if first_direction:
                    first_direction = False
                    restart = True
                    continue
                else:
                    # End of strip
                    break

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


def split_strips(strips):
    line_segments = []

    for strip in strips:
        
        strip_length = strip.shape[0]
        for idx in range(strip.shape[0]):
            if strip[idx][0] == 0:
                strip_length = idx
                break

        start_idx = 0

        counter = 0
        counter_stop = 4

        while True:
            line = strip[start_idx:]

            if line.shape[0] < 3:
                break

            if counter == counter_stop:
                print("start", line[0])

            vx, vy = 0, 0

            d = np.zeros(2)
            v = np.zeros(2)

            for idx in range(1, line.shape[0]):
                d[0] = line[idx][0] - line[0][0]
                d[1] = line[idx][1] - line[0][1]
                d = d / np.linalg.norm(d)

                v = (v * idx + d) / (idx + 1)
                v = v / np.linalg.norm(v)

                ang = (np.dot(d, v)) * 100

                if counter == counter_stop:
                    print(idx, line[idx], d, v, ang)

                if ang < 98:
                    line = line[0:idx]
                    line_segments.append(line)
                    print("done", start_idx, start_idx + idx - 1)
                    start_idx += idx
                    if counter == counter_stop:
                        print("hm")
                        quit()
                    counter += 1
                    break

        print("===")
        #print(line, strip_length)

        quit()




strips = extract_strips(image)
line_segments = split_strips(strips)

print(strips)
quit()

skimage.io.imshow(image)
skimage.io.show()


