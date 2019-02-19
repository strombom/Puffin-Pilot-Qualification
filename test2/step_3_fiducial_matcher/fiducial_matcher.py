
import cv2

from .point_extraction import extract_points
from .clusters import make_clusters
from .corners import make_corners
from .frames import match_corners


import os
import numpy as np


class FiducialMatcher:
    def __init__(self):
        pass

    def process(self, image, img_scale = 2):
        #from seaborn import color_palette
        #cv2.imwrite('img_1_fiducials.png', (image*255).astype(np.uint8))

        # Resize image, we lose precision but gain > 4x speed in the point extraction.
        #  cv2.INTER_NEAREST - 0.1 ms
        #  cv2.INTER_LINEAR  - 0.5 ms
        image = cv2.resize(src = image,
                           dsize = (image.shape[1] // img_scale, image.shape[0] // img_scale),
                           interpolation = cv2.INTER_NEAREST)


        #cv2.imwrite('img_2_fiducials_small.png', (image*255).astype(np.uint8))


        # Extract ideally 40 centroids from the raw image. If there are more than 100 points 
        #  there is something wrong and we don't want to waste time processing a bad image.
        points = extract_points(image, max_points = 1000)
        if len(points) < 10:
            return []
        points *= img_scale

        """
        cv2.imwrite('img_3_fiducials_small_extracted.png', (image*255).astype(np.uint8))

        palette = color_palette("bright", 1)
        source_path = os.path.dirname(os.path.abspath(__file__))
        image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
        image = cv2.imread(image_filepath)
        color = palette[0]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        for point in points:
            cv2.circle(image, tuple(point.astype(np.int64)), 3, color, -1)
        cv2.imwrite('img_4_points.png', image)
        """

        # Group the centroids into ideally 4 corner groups. We need at least two well formed
        #  corners to estimate the flying region.
        clusters = make_clusters(points)

        """
        palette = color_palette("bright", len(clusters))
        source_path = os.path.dirname(os.path.abspath(__file__))
        image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
        image = cv2.imread(image_filepath)
        for idx, cluster in enumerate(clusters):
            color = palette[idx]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            for point in cluster.points:
                cv2.circle(image, tuple(point.astype(np.int64)), 3, color, -1)
        cv2.imwrite('img_4_clusters.png', image)
        #print(points)
        #quit()
        """

        # Corners consists of ideally two perpendicular lines. In worst case a corner can
        #  consist of a single point.
        corners = make_corners(clusters)

        # Make frames from matching corners
        frames = match_corners(corners)

        return frames


if __name__ == '__main__':
    import os
    import pickle
    import numpy as np

    source_path = os.path.dirname(os.path.abspath(__file__))
    fiducials_path = os.path.join(source_path, '../step_2_fiducial_finder/fiducials.pickle')
    with open(fiducials_path, 'rb') as f:
        ficucials_image = pickle.load(f)

    #image = cv2.imread(image_path)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = image.astype(np.float32) / 255.0

    fiducial_matcher = FiducialMatcher()
    frames = fiducial_matcher.process(ficucials_image)

    print(frames)
    import pickle
    frames_path = os.path.join(source_path, 'frames.pickle')
    with open(frames_path, 'wb') as f:
        pickle.dump(frames, f)

    """
    import os
    import cv2
    from seaborn import color_palette
    palette = color_palette("bright", len(frames))
    source_path = os.path.dirname(os.path.abspath(__file__))
    image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
    image = cv2.imread(image_filepath)
    for frame_idx, frame in enumerate(frames):
        color = palette[frame_idx]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        for corner in frame.corners:
            for i in range(2):
                if not corner.has_lines:
                    for j in range(corner.points_count[i]):
                        point = corner.points[i][j].astype(np.int64)
                        cv2.circle(image, tuple(point), 3, color, -1)
                elif corner.has_lines:
                    for j in range(corner.points_count[i]):
                        point = corner.points[i][j].astype(np.int64)
                        cv2.circle(image, tuple(point), 3, color, -1)
                    line = corner.lines[i].astype(np.int64)
                    cv2.line(image, tuple(line[0]), tuple(line[1]), color, 2)
    cv2.imwrite('img_frames.png', image)
    """


    """
    from seaborn import color_palette
    palette = color_palette("bright", 1)
    source_path = os.path.dirname(os.path.abspath(__file__))
    image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
    image = cv2.imread(image_filepath)
    color = palette[0]
    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    for point in points:
        cv2.circle(image, tuple(point.astype(np.int64)), 3, color, -1)
    cv2.imwrite('img_frames.png', image)
    print(points)
    quit()
    """
