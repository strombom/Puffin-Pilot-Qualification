

import cv2
import skimage
import skimage.morphology

import numpy as np



image = cv2.imread('img_in.png', 0)
ret, image = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
image = (image / 255).astype(np.uint8)

#cv2.imwrite('img_out.png', image)


fid_size = 13
max_steps = 6
min_area = 50

# Add frame around image
h, w = image.shape
canvas = np.zeros((h+fid_size*2, w+fid_size*2), dtype=np.uint8)
canvas[fid_size:h+fid_size,fid_size:w+fid_size] = image
image = canvas
print(image.shape)

def get_section(x, y):
    return image[y-max_steps:y+max_steps+1,x-max_steps:x+max_steps+1]

def get_section_area(x, y):
    return np.sum(image[y-max_steps:y+max_steps+1,x-max_steps:x+max_steps+1])

points = []

for y in range(fid_size, image.shape[0] - fid_size):
    for x in range(fid_size, image.shape[1] - fid_size):
        if image[y][x] == 0:
            continue

        px, py = x, y
        for step in range(max_steps):
            nx, ny = px, py
            s = get_section_area(nx, ny)
            for dx in (-1, 0, 1):
                ns = get_section_area(nx + dx, ny + 1)
                if ns > s:
                    s = ns
                    nx = px + dx
                    ny = py + 1
            if ny == py:
                break
            px, py = nx, ny

        section = get_section(px, py)

        # Get center of point
        if np.sum(section) > min_area:
            m = cv2.moments(section)
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            # Compensate for image border, segment center and segment offset
            cx, cy = cx + px - max_steps * 3, cy + py - max_steps * 3
            points.append((cx, cy))

        # Remove blob from image
        section[:] = 0



im2 = cv2.imread('img_in.png', 0)
for point in points:
    x = int(point[0])
    y = int(point[1])

    im2[y-1][x] = 0
    im2[y][x-1] = 200
    im2[y][x] = 200
    im2[y][x+1] = 200
    im2[y+1][x] = 200
cv2.imwrite("out.png", im2)


print(len(points))
quit()





print(image)

