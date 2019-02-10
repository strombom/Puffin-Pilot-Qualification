
import cv2

from point_extraction import extract_points
from clusters import make_clusters
from corners import make_corners


def match_fiducials(img, img_scale = 2):
    # Resize image, we lose precision but gain > 4x speed in the point extraction.
    #  cv2.INTER_NEAREST - 0.1 ms
    #  cv2.INTER_LINEAR  - 0.5 ms
    img = cv2.resize(src = img,
                     dsize = (img.shape[1] / img_scale, img.shape[0] / img_scale),
                     interpolation = cv2.INTER_NEAREST)

    # Extract ideally 40 centroids from the raw image. If there are more than 100 points 
    #  there is something wrong and we don't want to waste time processing a bad image.
    points = extract_points(img, max_points = 1000)
    if len(points) < 10:
        return None
    points *= img_scale

    # Group the centroids into ideally 4 corner groups. We need at least two well formed
    #  corners to estimate the flying region.
    clusters = make_clusters(points)

    # Corners consists of ideally two perpendicular lines. In worst case a corner can
    #  consist of a single point.
    corners = make_corners(clusters)


    for corner in corners:
        print("---")
        for point in cluster:
            print(point)

    quit()


    return clusters

    if len(clusters) < 2:
        return None



if __name__ == '__main__':

    img = cv2.imread('img_in.png', cv2.IMREAD_GRAYSCALE)

    quadrilateral = match_fiducials(img)









