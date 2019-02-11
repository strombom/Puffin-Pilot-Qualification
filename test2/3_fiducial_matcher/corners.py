
import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64

from clusters import MAX_POINTS
from common import norm
from common import has_common_point
from common import make_line_from_points
from common import get_closest_point
from common import get_farthest_point_from_point
from common import get_farthest_point_from_line
from common import get_points_close_to_line
from common import line_intersection_angle


def make_corners(clusters):
    return jit_make_corners(clusters)


#@njit
def jit_make_corners(clusters):
    corners = []
    for c_idx in range(len(clusters)):
        corner = Corner(clusters[c_idx])
        corners.append(corner)

    return corners


class MatchingCriterion(IntEnum):
    NONE   = 1
    POINTS = 2
    LINE   = 3

corner_spec = [
    ('matching_criteria',      int64[:]),
    ('matching_points',        float64[:,:,:]),
    ('matching_points_count',  int64[:]),
    ('matching_lines',         float64[:,:,:]),
    ('matching_score',         float64)
]

@jitclass(corner_spec)
class Corner(object):
    # A corner has two sides, here called matching features 
    #  that are used to match two corners together.
    #  Example: A point in one corner can be matched with a 
    #           line of another corner.
    #
    # The corners will be sorted by matching_score when
    #  they are later matched together as quadilaterals.
    #

    def __init__(self, cluster):
        self.matching_criteria     = np.array((MatchingCriterion.NONE, MatchingCriterion.NONE), dtype=np.int64)
        self.matching_points       = np.empty((2, MAX_POINTS, 2))
        self.matching_points_count = np.zeros((2), dtype=np.int64)
        self.matching_lines        = np.empty((2, 2, 2))
        self.matching_score        = 0.0

        self._fit_lines(cluster)

    def _fit_lines(self, cluster):
        if cluster.points_count < 3:
            self.matching_criteria[:] = (MatchingCriterion.POINTS, MatchingCriterion.POINTS)
            self.matching_points[0][0] = cluster.points[0]
            self.matching_points[0][1] = cluster.points[0]
            self.matching_points_count[:] = (1, 1)
            self.matching_score = cluster.points_count
            return

        # More than two points
        self.matching_score = 3.0

        # Get the three points in the cluster that are
        #  as far away from eachother as possible.
        #  Those will serve as end points for potential lines.
        cog           = cluster.get_center_of_gravity()
        end_points    = np.empty((3, 2))
        end_points[0] = get_farthest_point_from_point(cluster.points, cluster.points_count, cog)
        end_points[1] = get_farthest_point_from_point(cluster.points, cluster.points_count, end_points[0])
        end_points[2] = get_farthest_point_from_line (cluster.points, cluster.points_count, end_points[0], end_points[1])

        # Draw lines between closest points
        lines = np.empty((6, 2, 2))
        for idx in range(3):
            lines[idx][0] = end_points[idx]
            lines[idx][1] = get_closest_point(cluster.points, cluster.points_count, end_points[idx])

        # Draw lines between end points
        for i in range(3):
            for j in range(i + 1, 3):
                idx = i + j - 1
                lines[idx+3][0] = end_points[i]
                lines[idx+3][1] = end_points[j]

        # Get all points near respective lines
        line_points       = np.empty((lines.shape[0], MAX_POINTS, 2))
        line_points_count = np.zeros((lines.shape[0]), dtype=np.int64)
        for i in range(lines.shape[0]):
            get_points_close_to_line(cluster, lines[i], line_points[i], line_points_count[i:i+1])

        # Sort the lines by point count
        fitness_idx       = np.argsort(line_points_count)[::-1]
        line_points       = line_points[fitness_idx]
        line_points_count = line_points_count[fitness_idx]

        if line_points_count[0] == 2:
            # The best line has only 2 points
            self.matching_criteria[0]     = MatchingCriterion.POINTS
            self.matching_points[0]       = line_points[0]
            self.matching_points_count[0] = line_points_count[0]

            # Set the second matching feature as the remaining point
            self.matching_criteria[1] = MatchingCriterion.POINTS
            self.matching_points[1][0] = get_farthest_point_from_line(points       = end_points, 
                                                                      points_count = 3, 
                                                                      p1           = line_points[0][0], 
                                                                      p2           = line_points[0][1])
            self.matching_points_count[1] = 1
            return

        # The best lines has at least 3 points, make a line
        first_feature_line = make_line_from_points(line_points[0], line_points_count[0])

        # Find second matching feature
        for i in range(0, lines.shape[0]):

            second_feature_line = make_line_from_points(line_points[i], line_points_count[i])

            # Check that the angle between the corner sides are sufficiently large for a proper corner
            angle = abs(line_intersection_angle(first_feature_line, second_feature_line))
            if angle > math.radians(20.0) and angle < math.radians(180.0 - 20.0):

                # The best lines has at least 3 points, make it a line
                self.matching_criteria[0]     = MatchingCriterion.LINE
                self.matching_lines[0]        = first_feature_line
                self.matching_points[0]       = line_points[0]
                self.matching_points_count[0] = line_points_count[0]
                self.matching_score += 1

                if line_points_count[1] == 2:  
                    if not has_common_point(line_points[0], line_points_count[0], line_points[i], line_points_count[i]):
                        # No common point found, set the second matching feature as a point feature
                        self.matching_criteria[1]     = MatchingCriterion.POINTS
                        self.matching_points[1]       = line_points[i]
                        self.matching_points_count[1] = line_points_count[i]
                        return

                # We can now assume that the second matching feature is a line
                self.matching_criteria[1]     = MatchingCriterion.LINE
                self.matching_lines[1]        = second_feature_line
                self.matching_points[1]       = line_points[i]
                self.matching_points_count[1] = line_points_count[i]
                self.matching_score += 1
                return

        # No good second feature was found, all points are probably in a straight line.
        #  We take the extreme points and put them in one feature each.
        self.matching_criteria[0]     = MatchingCriterion.POINTS
        self.matching_points[0][0]    = end_points[0]
        self.matching_points_count[0] = 1

        self.matching_criteria[1]     = MatchingCriterion.POINTS
        self.matching_points[1][0]    = end_points[1]
        self.matching_points_count[1] = 1
        return

        """
        print("first line")
        print(np.round(self.matching_lines[0], 1))
        print(np.round(self.matching_points[0][0:self.matching_points_count[0]], 1))
        print(self.matching_points_count[0])
        print("second line")
        #print(np.round(self.matching_lines[1], 1))
        print(np.round(self.matching_points[1][0:self.matching_points_count[1]], 1))
        print(self.matching_points_count[1])
        print("---")
        """

