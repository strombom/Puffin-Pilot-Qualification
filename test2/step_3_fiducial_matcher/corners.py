
from __future__ import print_function

import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import boolean, int64, float64

from .clusters import MAX_POINTS
from .common import norm, line_length, find_all_lines
#from .common import get_points_close_to_line
from .common import get_point_farthest_from_points
from .common import line_to_line_angle, line_to_point_angle, point_to_point_distance


def make_corners(clusters):
    #for cluster in clusters:
    #    print("===")
    #    print(cluster.points[0:cluster.points_count])
    #quit()
    return jit_make_corners(clusters)


#@njit
def jit_make_corners(clusters):
    corners = []
    for c_idx in range(len(clusters)):
        corner = Corner(clusters[c_idx].points, clusters[c_idx].points_count)
        corners.append(corner)
    return corners


class MatchingCriterion(IntEnum):
    NONE   = 1
    POINTS = 2
    LINES  = 3

corner_spec = [
    ('matching_criterion',     int64),
    ('matching_points',        float64[:,:,:]),
    ('matching_points_count',  int64[:]),
    ('matching_lines',         float64[:,:,:]),
    ('matching_angle_tol',     float64[:])
]

@jitclass(corner_spec)
class Corner(object):
    # A corner has two sides, here called matching features, 
    #  that are used to match two corners together.
    #
    # Example: A point in one corner can be matched with a 
    #          line of another corner.
    #

    def __init__(self, cluster_points, cluster_points_count):
        self.matching_criterion    = MatchingCriterion.NONE
        self.matching_points       = np.empty((2, MAX_POINTS, 2), dtype=np.float64)
        self.matching_points_count = np.zeros((2), dtype=np.int64)
        self.matching_lines        = np.empty((2, 2, 2), dtype=np.float64)
        self.matching_angle_tol    = np.zeros((2), dtype=np.float64)

        self._fit_lines(cluster_points, cluster_points_count)
        self._make_clockwise()

    def match(self, corner):
        # Test if this corner fits together with another corner

        # Line - Line
        if self.  matching_criterion == MatchingCriterion.LINES and \
           corner.matching_criterion == MatchingCriterion.LINES:
            angle_0 = line_to_point_angle(self.matching_lines[1],   corner.matching_lines[0][0])
            angle_1 = line_to_point_angle(corner.matching_lines[0], self.matching_lines[1][0])
            #print("match", angle_0, angle_1)
            if angle_0 < self.matching_angle_tol[1] and angle_1 < corner.matching_angle_tol[0]:
                return True
            return False

        # Line - Point
        if self.  matching_criterion == MatchingCriterion.LINES and \
           corner.matching_criterion == MatchingCriterion.POINTS:
            angle_1 = line_to_point_angle(self.matching_lines[1], corner.matching_points[0][0])
            angle_2 = line_to_point_angle(self.matching_lines[1], corner.matching_points[1][0])

            #print("line-point", angle_1, angle_2, self.matching_angle_tol)
            if angle_1 < self.matching_angle_tol[1] or angle_2 < self.matching_angle_tol[1]:
                return True

        # Point - Line
        if self.  matching_criterion == MatchingCriterion.POINTS and \
           corner.matching_criterion == MatchingCriterion.LINES:
            angle_1 = line_to_point_angle(corner.matching_lines[0], self.matching_points[0][0])
            angle_2 = line_to_point_angle(corner.matching_lines[0], self.matching_points[1][0])
            #print("point-line", angle_1, angle_2, corner.matching_angle_tol)
            if angle_1 < corner.matching_angle_tol[0] or angle_2 < corner.matching_angle_tol[0]:
                return True

        # Point - Point                
        if self.  matching_criterion == MatchingCriterion.POINTS and \
           corner.matching_criterion == MatchingCriterion.POINTS:
            # Two points can always be matched since there is no geometric constraint
            return True

        # No match found
        return False

    def get_distance(self, corner):
        if self.  matching_criterion == MatchingCriterion.LINES and \
           corner.matching_criterion == MatchingCriterion.LINES:
            p0 = self.  matching_lines[1][0]
            p1 = corner.matching_lines[0][0]
            return norm(p1 - p0)

        elif self.  matching_criterion == MatchingCriterion.POINTS and \
             corner.matching_criterion == MatchingCriterion.POINTS:
            p0 = self.  matching_points[1][0]
            p1 = corner.matching_points[0][0]
            return norm(p1 - p0)
        
        elif self.  matching_criterion == MatchingCriterion.LINES and \
             corner.matching_criterion == MatchingCriterion.POINTS:
            p0  = self.  matching_points[1][0]
            p1a = corner.matching_points[0][0]
            p1b = corner.matching_points[1][0]
            return min(norm(p1a - p0), norm(p1b - p0))
            
        elif self.  matching_criterion == MatchingCriterion.POINTS and \
             corner.matching_criterion == MatchingCriterion.LINES:
            p0a = self.  matching_points[0][0]
            p0b = self.  matching_points[1][0]
            p1  = corner.matching_points[0][0]
            return min(norm(p1 - p0a), norm(p1 - p0b))

    def _swap_matching_features(self):
        self.matching_points       = self.matching_points[::-1]
        self.matching_points_count = self.matching_criterion[::-1]
        self.matching_lines        = self.matching_lines[::-1]

    def test_line_pair_clockwise(self, line_pair):
        v1 = line_pair[0][1] - line_pair[0][0]
        v2 = line_pair[1][1] - line_pair[1][0]
        theta1 = math.atan2(v1[1], v1[0])
        theta2 = math.atan2(v2[1], v2[0])
        delta_theta = theta2 - theta1
        if delta_theta < 0:
            delta_theta += math.pi * 2
        if delta_theta < math.pi:
            # Swap line positions
            return False
        return True

    def _make_clockwise(self):
        if self.matching_criterion != MatchingCriterion.LINES:
            return

        # Make sure that both lines start in a common origin and point outwards
        origo_idx = (0, 0)
        min_distance = 1e9
        for idx in range(4):
            i, j = idx // 2, idx % 2
            distance = norm(self.matching_lines[0][i] - self.matching_lines[1][j])
            if distance < min_distance:
                min_distance = distance
                origo_idx = (i, j)
        
        for line_idx in range(2):
            if origo_idx[line_idx] != 0:
                self.matching_lines[line_idx] = self.matching_lines[line_idx][::-1]

        # Make lines clockwise
        if not self.test_line_pair_clockwise(self.matching_lines):
            self.matching_lines[:]        = self.matching_lines[::-1]
            self.matching_points[:]       = self.matching_points[::-1]
            self.matching_points_count[:] = self.matching_points_count[::-1]

    def _fit_lines(self, cluster_points, cluster_points_count):
        if cluster_points_count < 3:
            # Only 1 or 2 points in the cluster, no need to look for lines
            self.matching_criterion = MatchingCriterion.POINTS
            self.matching_points[0][0] = cluster_points[0]
            if cluster_points_count == 1:
                self.matching_points[1][0] = cluster_points[0]
                self.matching_points_count[:] = (1, 1)
            elif cluster_points_count == 2:
                self.matching_points[1][0] = cluster_points[1]
                self.matching_points_count[:] = (1, 1)
            else:
                self.matching_points[0][1] = cluster_points[1]
                self.matching_points[1][0] = cluster_points[2]
                self.matching_points_count[:] = (2, 1)
            return

        line = np.empty((2, 2))

        # Find the best line
        line_pair = np.zeros((2, MAX_POINTS, 2), dtype=np.uint64)
        line_pair_size = np.zeros((2), dtype=np.uint64)
        find_all_lines(cluster_points, cluster_points_count, line_pair, line_pair_size)

        if line_pair_size[1] > 0:
            # Two good lines found
            self.matching_criterion = MatchingCriterion.LINES
            for i in range(2):
                self.matching_lines[i][0]                         = line_pair[i][0]
                self.matching_lines[i][1]                         = line_pair[i][int(line_pair_size[i]-1)]
                self.matching_points[i][0:int(line_pair_size[i])] = line_pair[i][0:int(line_pair_size[i])]
                self.matching_points_count[i]                     = line_pair_size[i]

                # Set angle tolerance for matching. Long lines have better precisions and therefore lower tolerance.
                line_len = norm(self.matching_lines[i][1] - self.matching_lines[i][0])
                tolerance = max(math.radians(40 - 0.25 * line_len), math.radians(5))
                self.matching_angle_tol[i] = tolerance
            return

        if line_pair_size[0] > 0:
            self.matching_criterion       = MatchingCriterion.POINTS
            self.matching_points[0][0]    = line_pair[0][0]
            self.matching_points_count[0] = 1
            self.matching_points[1][0]    = line_pair[0][0]
            self.matching_points_count[1] = 1
            return

        self.matching_criterion       = MatchingCriterion.POINTS
        self.matching_points[0][0:cluster_points_count] = cluster_points[0:cluster_points_count]
        self.matching_points_count[0] = cluster_points_count
        self.matching_points[1][0]    = cluster_points[0]
        self.matching_points_count[1] = 1
        return
