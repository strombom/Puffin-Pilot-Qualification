
from __future__ import print_function

import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64, boolean

from corners import MatchingCriterion


def match_corners(corners, debug = False):

    # Add all corners to frames, make new frames when required
    # Frames are lists of corners
    frames = []
    point_corners = []
    for corner in corners:
        if corner.matching_criterion == MatchingCriterion.LINES:
            frames.append([corner])
        elif corner.matching_criterion == MatchingCriterion.POINTS:
            point_corners.append([corner])

    merge_frames(frames)
    merge_point_corners(frames, point_corners)

    # Remove frames with only points and frames with only one corner
    for i in range(len(frames)-1, -1, -1):
        frame = frames[i]
        if len(frame) == 1:
            del frames[i]
            continue
        has_lines = False
        for corner in frame:
            if corner.matching_criterion == MatchingCriterion.LINES:
                has_lines = True
                break
        if not has_lines:
            del frames[i]

    import os
    import cv2
    from seaborn import color_palette
    palette = color_palette("bright", 4)
    source_path = os.path.dirname(os.path.abspath(__file__))
    image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
    image = cv2.imread(image_filepath)
    for frame_idx, frame in enumerate(frames):
        for corner_idx, corner in enumerate(frame):
            color = palette[corner_idx]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            for i in range(2):
                if corner.matching_criterion == MatchingCriterion.POINTS:
                    for j in range(corner.matching_points_count[i]):
                        point = corner.matching_points[i][j].astype(np.int64)
                        #print("point")
                        #print(tuple(point))
                        cv2.circle(image, tuple(point), 2, color, -1)
                elif corner.matching_criterion == MatchingCriterion.LINES:

                    for j in range(corner.matching_points_count[i]):
                        point = corner.matching_points[i][j].astype(np.int64)
                        #print("point")
                        #print(tuple(point))
                        cv2.circle(image, tuple(point), 2, color, -1)
                    line = corner.matching_lines[i].astype(np.int64)
                    #print("line")
                    #print(tuple(line[0]))
                    cv2.line(image, tuple(line[0]), tuple(line[-1]), color, 1)

    print("write frames")
    cv2.imwrite('img_5_frames.png', image)

    

    # Make frames
    for idx, frame in enumerate(frames):
        frames[idx] = Frame(frame)

    return frames


class Frame:
    def __init__(self, corners):
        self.corners = []
        for corner in corners:
            self.corners.append(FrameCorner(corner))

class FrameCorner:
    def __init__(self, corner):
        if corner.matching_criterion == MatchingCriterion.LINES:
            self.has_lines = True
        else:
            self.has_lines = False

        self.points = corner.matching_points
        self.points_count = corner.matching_points_count
        self.lines = corner.matching_lines


def merge_point_corners(frames, point_corners):
    # Make sure that frames with many corners have highest merging priority
    frames.sort(key=len, reverse=True)
    frames.extend(point_corners)
    merge_frames(frames)


def merge_frames(frames):
    best_matches = np.zeros(2, np.int64)
    matches      = np.zeros(2, np.int64)

    print("start")
    print(frames)

    # Merge frames
    idx_i = 0
    while idx_i < len(frames):
        if len(frames[idx_i]) == 4:
            idx_i += 1
            continue

        best_distance = 1e9
        best_fitness_idx = -1

        # Find closest match
        for idx_j in range(idx_i + 1, len(frames)):
            distance = merging_fitness(frames[idx_i], frames[idx_j], matches)
            if distance < 0:
                continue
            if distance < best_distance:
                best_distance = distance
                best_fitness_idx = idx_j
                best_matches[:] = matches

        if best_fitness_idx >= 0:
            if best_matches[0]:
                for idx, corner in enumerate(frames[best_fitness_idx]):
                    frames[idx_i].insert(idx, corner)
                    print("insert")
            else:
                frames[idx_i].extend(frames[best_fitness_idx])
                print("extend")

            del frames[best_fitness_idx]
            idx_i -= 1
        idx_i += 1

    print("end")
    print(frames)
    #quit()


def merge_point_corners(frames, point_corners):
    # Make sure that frames with many corners have highest merging priority
    frames.sort(key=len, reverse=True)
    frames.extend(point_corners)
    merge_frames(frames)


#@njit
def merging_fitness(frame1, frame2, matches):
    total_corners = len(frame1) + len(frame2)
    print("total_corners", total_corners)

    if total_corners > 4:
        return -1

    if frame1[-1].match(frame2[0]):
        matches[1] = 1
    else:
        matches[1] = 0

    if frame2[-1].match(frame1[0]):
        matches[0] = 1
    else:
        matches[0] = 0

    if not matches[0] and not matches[1]:
        return -1

    if total_corners == 4:
        if not matches[0] or not matches[1]:
            return -1

    if matches[0]:
        return frame1[0].get_distance(frame2[-1])
    else:
        return frame2[0].get_distance(frame1[-1])






"""


    import cv2
    from seaborn import color_palette
    palette = color_palette("bright", len(frames))
    image_filepath = 'img_in.png'
    image = cv2.imread(image_filepath)
    for frame_idx, frame in enumerate(frames):
        color = palette[frame_idx]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        for corner in frame:
            for i in range(2):
                if corner.matching_criterion == MatchingCriterion.POINTS:
                    for j in range(corner.matching_points_count[i]):
                        point = corner.matching_points[i][j].astype(np.int64)
                        #print("point")
                        #print(tuple(point))
                        cv2.circle(image, tuple(point), 3, color, -1)
                elif corner.matching_criterion == MatchingCriterion.LINES:

                    for j in range(corner.matching_points_count[i]):
                        point = corner.matching_points[i][j].astype(np.int64)
                        #print("point")
                        #print(tuple(point))
                        cv2.circle(image, tuple(point), 3, color, -1)
                    line = corner.matching_lines[i].astype(np.int64)
                    #print("line")
                    #print(tuple(line[0]))
                    cv2.line(image, tuple(line[0]), tuple(line[1]), color, 2)

    cv2.imwrite('img_frames.png', image)
    quit()

"""

#print(frames)



#for corner in corners:
#    print("---", corner.matching_score)

#for corner in cluster:
#    print(point)


"""
import cv2
image_filepath = 'img_in.png'
image = cv2.imread(image_filepath)
colors = ((255,0,0), (0,255,0))
for corner in corners:
    print("---")
    for i in range(2):
        if corner.matching_criterion[i] == MatchingCriterion.POINTS:
            for j in range(corner.matching_points_count[i]):
                point = corner.matching_points[i][j].astype(np.int64)
                print("point")
                print(tuple(point))
                cv2.circle(image, tuple(point), 3, colors[i], -1)
        elif corner.matching_criterion[i] == MatchingCriterion.LINE:

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

"""
import cv2
from seaborn import color_palette
palette = color_palette("bright", 2)
image_filepath = 'img_in.png'
image = cv2.imread(image_filepath)

colors = []
for i in range(2):
    color = palette[i]
    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
    colors.append(color)

for corner in corners:
    for i in range(2):
        if corner.matching_criterion == MatchingCriterion.POINTS:
            for j in range(corner.matching_points_count[i]):
                point = corner.matching_points[i][j].astype(np.int64)
                #print("point")
                #print(tuple(point))
                cv2.circle(image, tuple(point), 3, colors[i], -1)

        elif corner.matching_criterion == MatchingCriterion.LINE:
            for j in range(corner.matching_points_count[i]):
                point = corner.matching_points[i][j].astype(np.int64)
                #print("point")
                #print(tuple(point))
                cv2.circle(image, tuple(point), 3, colors[i], -1)
            line = corner.matching_lines[i].astype(np.int64)
            #print("line")
            #print(tuple(line[0]))
            cv2.line(image, tuple(line[0]), tuple(line[1]), colors[i], 2)

            cv2.circle(image, tuple(line[0]), 5, colors[i], -1)

cv2.imwrite('img_corners.png', image)
quit()
"""
