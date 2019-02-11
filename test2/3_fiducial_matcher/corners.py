
import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import int64, float64

from clusters import MAX_POINTS
from common import norm
from common import make_line_from_points
from common import get_farthest_point_from_point
from common import get_farthest_point_from_line
from common import get_points_close_to_line


def make_corners(clusters):
    return jit_make_corners(clusters)



class MatchingCriterion(IntEnum):
    POINTS = 2
    LINE   = 3

corner_spec = [
    ('matching_criteria',      int64[:]),
    ('matching_points',        float64[:,:,:]),
    ('matching_points_count'), int64[:]
    ('matching_lines',         float64[:,:]),
    ('fitting_score',          float64)
]

#@jitclass(corner_spec)
class Corner(object):
    # A corner has two sides, here called matching features 
    #  that are used to match two corners together.
    #  Example: A point in one corner can be matched with a 
    #           line of another corner.

    def __init__(self, cluster):
        self.matching_criteria     = np.array((MatchingCriterion.POINTS, MatchingCriterion.POINTS), dtype=np.int64)
        self.matching_points       = np.empty((2, MAX_POINTS, 2))
        self.matching_points_count = np.zeros((2), dtype=np.int64)
        self.matching_lines        = np.empty((2, 2, 2))
        self.fitting_score         = 0.0

        self._fit_lines(cluster)

    def _fit_lines(self, cluster):
        if cluster.points_count == 1:
            self.matching_points[0][0] = cluster.points[0]
            self.matching_points[0][1] = cluster.points[0]
            self.matching_points_count[:] = (1, 1)
            self.fitting_score = 1.0
            return

        elif cluster.points_count == 2:
            self.matching_points[0][0] = cluster.points[0]
            self.matching_points[0][1] = cluster.points[1]
            self.matching_points_count[:] = (1, 1)
            self.fitting_score = 2.0
            return

        else:
            # More than two points
            self.fitting_score = 3.0

        # Get the three points in the cluster that are
        #  as far away from eachother as possible.
        #  Those will serve as end points for potential lines.
        cog           = cluster.get_center_of_gravity()
        end_points    = np.empty((3, 2))
        end_points[0] = get_farthest_point_from_point(cluster.points, cluster.points_count, cog)
        end_points[1] = get_farthest_point_from_point(cluster.points, cluster.points_count, end_points[0])
        end_points[2] = get_farthest_point_from_line (cluster.points, cluster.points_count, end_points[0], end_points[1])

        # Draw lines between edge points
        lines = np.empty((3, 2, 2))
        for i in range(3):
            for j in range(i + 1, 3):
                idx = i + j - 1
                lines[idx][0] = end_points[i]
                lines[idx][1] = end_points[j]

        # Get all points near respective lines
        line_points       = np.empty((3, MAX_POINTS, 2))
        line_points_count = np.zeros((3), dtype=np.int64)
        for i in range(3):
            get_points_close_to_line(cluster, lines[i], line_points[i], line_points_count[i:i+1])

        # Sort the lines by point count
        fitness_idx       = np.argsort(line_points_count)[::-1]
        line_points       = line_points[fitness_idx]
        line_points_count = line_points_count[fitness_idx]

        #print("===")
        #for i in range(3):
        #    print(np.round(line_points[i][0:line_points_count[i]], 1))
        #    print("---")

        # Always set the first matching feature as a line
        self.matching_criteria[0] = MatchingCriterion.LINE
        self.matching_lines[0] = make_line_from_points(line_points[0], line_points_count[0])
        self.matching_points[0] = line_points[0]
        self.matching_points_count[0] = line_points_count[0]

        if line_points_count[1] == 2:
            # We could not find a clear corner
            # Set the second matching feature as a point
            self.matching_criteria[1] = MatchingCriterion.POINTS
            self.matching_points[1][0] = get_farthest_point_from_line(points       = line_points[1], 
                                                                      points_count = line_points_count[1], 
                                                                      p1           = self.matching_lines[0][0], 
                                                                      p2           = self.matching_lines[0][1])
            self.matching_points_count[1] = 1

        else:
            # Set the second matching feature as a line
            self.matching_criteria[1] = MatchingCriterion.LINE
            self.matching_lines[1] = make_line_from_points(line_points[1], line_points_count[1])
            self.matching_points[1] = line_points[1]
            self.matching_points_count[1] = line_points_count[1]

            """
            print("first line")
            print(np.round(self.matching_lines[0], 1))
            print(np.round(self.matching_points[0][0:self.matching_points_count[0]], 1))
            print(self.matching_points_count[0])
            print("second line")
            print(np.round(self.matching_lines[1], 1))
            print(np.round(self.matching_points[1][0:self.matching_points_count[1]], 1))
            print(self.matching_points_count[1])
            print("---")
            print("best has only two")
            print(np.round(end_points, 1))
            print("---")
            print(np.round(cluster.points, 1))
            """


        # If first and second matching features have only 2 points
        #  check that their angle is at least 30 degrees, if they
        #  are in a straight line only keep the two farthest points.


#@njit
def jit_make_corners(clusters):
    corners = []
    for c_idx in range(len(clusters)):
        corner = Corner(clusters[c_idx])
        corners.append(corner)

    return corners


