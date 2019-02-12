
import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64

from corners import MatchingCriterion


def match_corners(corners):
    return jit_match_corners(corners)




class Frame:
    def __init__(self, corner):
        self.corners = [corner]

    def append(self, new_corner):
        if len(self.corners) == 4:
            return False

        if len(self.corners) == 3:
            # The frame has 3 corners, we must match the first and last corner
            #  so let's make sure the first corner is a match
            if not new_corner.match(self.corners[0], mark_matched=False):
                return False
        else:
            # The frame has 1 or 2 corners, check if the first corner is a match
            if new_corner.match(self.corners[0], mark_matched=True):
                self.corners.insert(new_corner, 0)
                return True

        # Check the last corner
        if new_corner.match(corner, mark_matched=True):
            self.corners.append(new_corner)
            return True
        else:
            return False


#@njit
def jit_match_corners(corners):

    # Sort corners by matching score
    corners = sorted(corners, key=lambda corner: corner.matching_score, reverse=True)

    # Add all corners to frames, make new frames when required
    frames = []
    for corner in corners:
        for frame in frames:
            if frame.append(corner):
                break
        else:
            # No suitable frame found, make a new frame for the corner
            frames.append(Frame(corner))

    # Merge frames

    print(frames)



    #for corner in corners:
    #    print("---", corner.matching_score)

    #for corner in cluster:
    #    print(point)

    print(len(frames))
    quit()




    """
    import cv2
    image_filepath = 'img_in.png'
    image = cv2.imread(image_filepath)
    colors = ((255,0,0), (0,255,0))
    for corner in corners:
        print("---")
        for i in range(2):
            if corner.matching_criteria[i] == MatchingCriterion.POINTS:
                for j in range(corner.matching_points_count[i]):
                    point = corner.matching_points[i][j].astype(np.int64)
                    print("point")
                    print(tuple(point))
                    cv2.circle(image, tuple(point), 3, colors[i], -1)
            elif corner.matching_criteria[i] == MatchingCriterion.LINE:

                for j in range(corner.matching_points_count[i]):
                    point = corner.matching_points[i][j].astype(np.int64)
                    print("point")
                    print(tuple(point))
                    cv2.circle(image, tuple(point), 3, colors[i], -1)
                line = corner.matching_lines[i].astype(np.int64)
                print("line")
                print(tuple(line[0]))
                cv2.line(image, tuple(line[0]), tuple(line[1]), colors[i], 2)
    cv2.imwrite('img_out.png', image)
    """

    quit()


    return []
