
from __future__ import print_function

import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64, boolean

from .common import point_to_point_distance
from .corners import MatchingCriterion
from .corners import corners_test_match, get_corner_distance


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
    pick_points(frames)

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

    merge_large_corners(frames)
    order_corners(frames)
    keep_furthest_point(frames)
    points_sanity_check(frames)

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
            else:
                frames[idx_i].extend(frames[best_fitness_idx])

            del frames[best_fitness_idx]
            idx_i -= 1
        idx_i += 1


def merge_large_corners(frames):
    large_corner_count = []
    corner_counts = []

    for frame in frames:
        count = 0
        for corner in frame:
            if corner.matching_points_count[0] + corner.matching_points_count[1] > 4:
                count += 1
        large_corner_count.append(count)

    if len(large_corner_count) > 1 and large_corner_count[0] == 3 and large_corner_count[1] > 0:

        if len(frames[0]) == 3:
            frames[0].append(None)
            small_idx = 3
        else:
            small_idx = 0
            small_size = 1e9
            for idx, corner in enumerate(frames[0]):
                size = corner.matching_points_count[0] + corner.matching_points_count[1]
                if size < small_size:
                    small_size = size
                    small_idx = idx

        big_idx = 0
        big_size = 0
        for idx, corner in enumerate(frames[1]):
            size = corner.matching_points_count[0] + corner.matching_points_count[1]
            if size > big_size:
                big_size = size
                big_idx = idx

        frames[0][small_idx] = frames[1][big_idx]
        del frames[1]

    elif len(large_corner_count) > 1 and large_corner_count[0] == 2 and large_corner_count[1] > 0:
        while large_corner_count[1] > 0:
            large_corner_count[1] -= 1
            if len(frames[0]) < 4:
                frames[0].append(None)
                small_idx = len(frames[0]) - 1
            else:
                small_idx = 0
                small_size = 1e9
                for idx, corner in enumerate(frames[0]):
                    size = corner.matching_points_count[0] + corner.matching_points_count[1]
                    if size < small_size:
                        small_size = size
                        small_idx = idx

            big_idx = 0
            big_size = 0
            for idx, corner in enumerate(frames[1]):
                size = corner.matching_points_count[0] + corner.matching_points_count[1]
                if size > big_size:
                    big_size = size
                    big_idx = idx

            frames[0][small_idx] = frames[1][big_idx]
            del frames[1][big_idx]

def order_corners(frames):
    for frame in frames:
        cpx = []
        cpy = []
        corners = []
        fmin_x, fmax_x, fmin_y, fmax_y = 1e9, 0, 1e9, 0
        for corner in frame:
            if corner is None:
                continue
            cx, cy = 0, 0
            for side in range(2):
                for pidx in range(corner.matching_points_count[side]):
                    cx = cx + corner.matching_points[side][pidx][0]
                    cy = cy + corner.matching_points[side][pidx][1]
            cx = cx / (corner.matching_points_count[0] + corner.matching_points_count[1])
            cy = cy / (corner.matching_points_count[0] + corner.matching_points_count[1])
            fmin_x, fmax_x = min(fmin_x, cx), max(fmax_x, cx)
            fmin_y, fmax_y = min(fmin_y, cy), max(fmax_y, cy)
            cpx.append(cx)
            cpy.append(cy)
            corners.append(corner)

        new_frame = [None, None, None, None]
        cx, cy = (fmin_x + fmax_x) / 2, (fmin_y + fmax_y) / 2    
        for idx in range(len(cpx)):
            if   cpx[idx] < cx and cpy[idx] < cy:
                new_frame[0] = corners[idx]
            elif cpx[idx] > cx and cpy[idx] < cy:
                new_frame[1] = corners[idx]
            elif cpx[idx] > cx and cpy[idx] > cy:
                new_frame[2] = corners[idx]
            elif cpx[idx] < cx and cpy[idx] > cy:
                new_frame[3] = corners[idx]

        frame.clear()
        for corner in new_frame:
            if corner is not None:
                frame.append(corner)


def keep_furthest_point(frames):
    for frame in frames:
        if len(frame) < 4:
            continue

        for corner_idx, corner in enumerate(frame):
            if corner is None:
                continue

            if corner.matching_criterion != MatchingCriterion.POINTS:
                continue

            ci, cj = (corner_idx - 1) % 4, (corner_idx + 1) % 4
            if frame[ci] is None or frame[cj] is None:
                continue

            point = np.zeros(2, np.float64)
            max_distance = 0

            p1, p2 = frame[ci].matching_points[0][0], frame[cj].matching_points[0][0]
            for side_this in range(2):
                for point_k in range(corner.matching_points_count[side_this]):
                    p0 = corner.matching_points[side_this][point_k]
                    d = point_to_point_distance(p0, p1) + point_to_point_distance(p0, p2)
                    if d > max_distance:
                        max_distance = d
                        point = p0

            corner.matching_points[0] = point
            corner.matching_points[1] = point
            corner.matching_points_count[0] = 1
            corner.matching_points_count[1] = 1

def points_sanity_check(frames):
    for frame in frames:
        if len(frame) < 4:
            continue

        if None in frame:
            continue

        cgs = np.zeros((4, 2), np.float64)
        for ci, corner in enumerate(frame):
            cgs[ci] =  np.sum(corner.matching_points[0][0:corner.matching_points_count[0]])
            cgs[ci] += np.sum(corner.matching_points[1][0:corner.matching_points_count[1]])
            cgs[ci] = cgs[ci] / (corner.matching_points_count[0] + corner.matching_points_count[1])

        for ci, corner in enumerate(frame):
            print("psc")


#@njit
def pick_points(frames):
    if len(frames) < 2:
        return

    if len(frames[0]) != 3:
        return

    min_x, max_x = 1e9, 0
    for corner_i in range(len(frames[0])):
        corner = frames[0][corner_i]
        #print("corner", corner)
        for side_j in range(2):
            for point_k in range(corner.matching_points_count[side_j]):
                point = corner.matching_points[side_j][point_k]
                min_x, max_x = min(min_x, point[0]), max(max_x, point[0])
    tolerance = (max_x - min_x) * 0.1
    left_x, right_x = min_x + tolerance, max_x - tolerance

    # Find picky points
    best_corner_idx = -1
    best_corner_distance = 1e9
    for corner_i in range(len(frames[1])):
        corner = frames[1][corner_i]
        #print("corner", corner)
        for side_j in range(2):
            for point_k in range(corner.matching_points_count[side_j]):
                point = corner.matching_points[side_j][point_k]
                distance = min(abs(left_x - point[0]), abs(right_x - point[0]))
                if distance < best_corner_distance:
                    best_corner_distance = distance
                    best_corner_idx = corner_i

    if best_corner_idx != -1:
        frames[0].append(frames[1][best_corner_idx])
        if len(frames[1]) == 1:
            del frames[1]
        else:
            del frames[1][best_corner_idx]


@njit
def merging_fitness(frame1, frame2, matches):
    total_corners = len(frame1) + len(frame2)

    if total_corners > 4:
        return -1

    if corners_test_match(frame1[-1], frame2[0]):
        matches[1] = 1
    else:
        matches[1] = 0

    if corners_test_match(frame2[-1], frame1[0]):
        matches[0] = 1
    else:
        matches[0] = 0

    if not matches[0] and not matches[1]:
        # No match
        return -1

    if total_corners == 4:
        if not matches[0] or not matches[1]:
            # Only one match, two required
            return -1

    if matches[0]:
        return get_corner_distance(frame1[0], frame2[-1])
    else:
        return get_corner_distance(frame2[0], frame1[-1])

