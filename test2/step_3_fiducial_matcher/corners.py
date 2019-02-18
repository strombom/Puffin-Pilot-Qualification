
from __future__ import print_function

import math
import numpy as np
from enum import IntEnum
from numba import jit, njit, jitclass
from numba import boolean, int64, float64

from clusters import MAX_POINTS
from common import norm, line_length, has_common_point, make_line_from_points
from common import get_closest_point, get_points_close_to_line
from common import get_point_farthest_from_points, get_point_farthest_from_line
from common import line_to_line_angle, line_to_point_angle, point_to_point_distance


def make_corners(clusters):
    return jit_make_corners(clusters)


#@njit
def jit_make_corners(clusters):
    corners = []
    for c_idx in range(len(clusters)):
        #if c_idx == 3:
        corner = Corner(clusters[c_idx])
        corners.append(corner)
    #quit()

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

#@jitclass(corner_spec)
class Corner(object):
    # A corner has two sides, here called matching features, 
    #  that are used to match two corners together.
    #
    # Example: A point in one corner can be matched with a 
    #          line of another corner.
    #

    def __init__(self, cluster):
        self.matching_criterion    = MatchingCriterion.NONE
        self.matching_points       = np.empty((2, MAX_POINTS, 2))
        self.matching_points_count = np.zeros((2), dtype=np.int64)
        self.matching_lines        = np.empty((2, 2, 2))
        self.matching_angle_tol    = np.zeros((2))

        self._fit_lines(cluster)
        self._make_clockwise()

    def match(self, corner):
        # Test if this corner fits together with another corner


        # Line - Line
        if self.  matching_criterion == MatchingCriterion.LINES and \
           corner.matching_criterion == MatchingCriterion.LINES:
            angle_0 = line_to_point_angle(self.matching_lines[1],   corner.matching_lines[0][0])
            angle_1 = line_to_point_angle(corner.matching_lines[0], self.matching_lines[1][0])
            print("match", angle_0, angle_1)
            if angle_0 < self.matching_angle_tol[1] and angle_1 < corner.matching_angle_tol[0]:
                return True
            return False

        # Line - Point
        if self.  matching_criterion == MatchingCriterion.LINES and \
           corner.matching_criterion == MatchingCriterion.POINTS:
            angle_1 = line_to_point_angle(self.matching_lines[1], corner.matching_points[0][0])
            angle_2 = line_to_point_angle(self.matching_lines[1], corner.matching_points[1][0])
            if angle_1 < corner.matching_angle_tol[1] or angle_2 < corner.matching_angle_tol[1]:
                return True

        # Point - Line
        if self.  matching_criterion == MatchingCriterion.POINTS and \
           corner.matching_criterion == MatchingCriterion.LINES:
            angle_1 = line_to_point_angle(corner.matching_lines[0], self.matching_points[0][0])
            angle_2 = line_to_point_angle(corner.matching_lines[0], self.matching_points[1][0])
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

    #@jit(nopython=True)
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
            distance = point_to_point_distance(self.matching_lines[0][i], self.matching_lines[1][j])
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


    def _fit_lines(self, cluster):
        if cluster.points_count < 3:
            # Only 1 or 2 points in the cluster, no need to look for lines
            self.matching_criterion = MatchingCriterion.POINTS
            self.matching_points[0][0] = cluster.points[0]
            if cluster.points_count == 1:
                self.matching_points[1][0] = cluster.points[0]
                self.matching_points_count[:] = (1, 1)
            elif cluster.points_count == 2:
                self.matching_points[1][0] = cluster.points[1]
                self.matching_points_count[:] = (1, 1)
            else:
                self.matching_points[0][1] = cluster.points[1]
                self.matching_points[1][0] = cluster.points[2]
                self.matching_points_count[:] = (2, 1)
            return

        from common import find_all_lines

        line = np.empty((2, 2))

        # Find the best line
        line_pair = np.zeros((2, cluster.points_count, 2), dtype=np.uint64)
        line_pair_size = find_all_lines(cluster.points, cluster.points_count, line_pair)

        if line_pair_size[1] > 1:
            # Two good lines found
            self.matching_criterion = MatchingCriterion.LINES
            for i in range(2):
                print(line_pair_size[i]-1)
                self.matching_lines[i][0]                    = line_pair[i][0]
                self.matching_lines[i][1]                    = line_pair[i][int(line_pair_size[i]-1)]
                self.matching_points[i][0:line_pair_size[i]] = line_pair[i][0:line_pair_size[i]]
                self.matching_points_count[i]                = line_pair_size[i]

                # Set angle tolerance for matching. Long lines have better precisions and therefore lower tolerance.
                self.matching_angle_tol[i] = max(math.radians(25 - 0.25 * line_length(self.matching_lines[i])), math.radians(5))

            print("fitlines done")
            return



        print("found lines", line_pair_size)
        quit()







        if best_count == 0:
            # Not a single line with at least 3 points could be found,
            #  very strange, return a single point
            self.matching_criterion = MatchingCriterion.POINTS
            self.matching_points[0][0] = cluster.points[0]
            self.matching_points[1][0] = cluster.points[0]
            self.matching_points_count[:] = (1, 1)
            return

        # Split points
        first_remaining = np.empty((points_count, 2))
        first_remaining_count = 0
        first_line = np.empty((points_count, 2))
        first_line_count = 0
        line[0], line[1] = points[first_pair[0]], points[first_pair[1]]
        for i in range(points_count):
            distance = line_to_point_distance(line, points[i])
            if distance < distance_threshold:
                first_line[first_line_count] = points[i]
                first_line_count += 1
            else:
                first_remaining[first_remaining_count] = points[i]
                first_remaining_count += 1

        if first_remaining_count < 2:
            # No points or only one point remains
            self.matching_criterion = MatchingCriterion.POINTS
            self.matching_points[0][0] = line[0]
            self.matching_points[1][0] = line[1]
            self.matching_points_count[:] = (1, 1)
            return

        candidate_lines_max_count = 10
        candidate_lines_count = 0
        candidate_lines = np.empty((candidate_lines_max_count, 2, 2))

        # Append first line to candidates
        candidate_lines_count = 1
        candidate_lines[0] = line


        candidate_lines_points[0] = first_line
        candidate_lines_points_count[0] = first_count

        if first_remaining_count == 2:
            # Two points remains
            for idx in range(4):
                i, j = idx % 2, idx // 2
                p0, p1 = first_remaining[j], first_remaining[1-j]
                line[0], line[1] = points[first_pair[i]], p0
                distance = line_to_point_distance(line, p1)
                if distance < distance_threshold:
                    # Good candidate found, also make sure that both 
                    #  points are closest to the current line point
                    lp0, lp1 = points[first_pair[i]], points[first_pair[1-i]]
                    if point_to_point_distance(p0, lp0) < point_to_point_distance(p0, lp1) and \
                       point_to_point_distance(p1, lp0) < point_to_point_distance(p1, lp1):
                        # Add to candidate lines


                        candidate_lines_count += 1




                        #self.matching_criterion = MatchingCriterion.LINES:

                        self.matching_criterion       = MatchingCriterion.LINES
                        self.matching_lines[0]        = first_feature_line
                        self.matching_points[0]       = line_points[0]
                        self.matching_points_count[0] = line_points_count[0]



                        self.matching_points[0][0] = first_remaining[0]
                        self.matching_points[1][0] = first_remaining[first_remaining_count-1]
                        self.matching_points_count[:] = (1, 1)







                    line_candidates[i*2+j][0] = points[first_pair[i]]
                    line_candidates[i*2+j][1] = first_remaining[j]
            line_candidates_count = 4








        # Find a second line
        second_pair = np.zeros(2, dtype=np.uint64)
        second_count = find_best_line(points, points_count, second_pair, distance_threshold)
        if second_count == 0:
            # No second line could be found, find a point close to the line ends
            d1 = distance_from_points_to_point(remaining, remaining_count, line[0])
            d2 = distance_from_points_to_point(remaining, remaining_count, line[1])
            self.matching_criterion = MatchingCriterion.POINTS
            if d1 < d2:
                self.matching_points[0][0] = line[0]
            else:
                self.matching_points[0][0] = line[1]
            self.matching_points[1][0] = self.matching_points[0][0]
            self.matching_points_count[:] = (1, 1)
            return

        # Split points
        second_remaining = np.empty((first_remaining_count, 2))
        second_remaining_count = 0
        second_line = np.empty((first_remaining_count, 2))
        second_line_count = 0
        line[0], line[1] = points[second_pair[0]], points[second_pair[1]]
        for i in range(first_remaining_count):
            distance = line_to_point_distance(line, points[i])
            if distance < distance_threshold:
                best_line[best_line_count] = points[i]
                best_line_count += 1
            else:
                second_remaining[second_remaining_count] = first_remaining[i]
                second_remaining_count += 1


        if second_remaining_count > 2:

            # Find a third line
            print("find third line")
            quit()



        if remaining_count == 0:
            print("remain 0")
            quit()

        if remaining_count == 1:
            print("remain 1")
            quit()

        #while remaining_count > 1:









        print("==")
        print(best_line[0:best_line_count])
        print("==")
        print(remaining[0:remaining_count])
        print("==")
        print(line)
        quit()






        # Get the three points in the cluster that are
        #  as far away from eachother as possible.
        #  Those will serve as end points for potential lines.

        lines = np.empty((2, 2, 2))
        line_finder(cluster.points, cluster.points_count, lines)

        cog           = cluster.get_center_of_gravity()
        end_points[0] = get_point_farthest_from_points(cluster.points, cluster.points_count, cog)
        end_points[1] = get_point_farthest_from_points(cluster.points, cluster.points_count, end_points[0])
        end_points[2] = get_point_farthest_from_line (cluster.points, cluster.points_count, end_points[0], end_points[1])

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
            self.matching_criterion       = MatchingCriterion.POINTS
            self.matching_points[0]       = line_points[0]
            self.matching_points_count[0] = line_points_count[0]

            # Set the second matching feature as the remaining point
            self.matching_points[1][0] = get_point_farthest_from_line(points       = end_points, 
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
            angle = abs(line_to_line_angle(first_feature_line, second_feature_line))
            if angle > math.radians(20.0) and angle < math.radians(180.0 - 20.0):

                # The best lines has at least 3 points, make it a line
                self.matching_criterion       = MatchingCriterion.LINES
                self.matching_lines[0]        = first_feature_line
                self.matching_points[0]       = line_points[0]
                self.matching_points_count[0] = line_points_count[0]

                # We can now assume that the second matching feature is a line
                self.matching_lines[1]        = second_feature_line
                self.matching_points[1]       = line_points[i]
                self.matching_points_count[1] = line_points_count[i]

                if line_points_count[1] == 2:
                    if not has_common_point(line_points[0], line_points_count[0], line_points[i], line_points_count[i]):
                        # No common point found, set the second matching feature as a point feature
                        point = get_point_farthest_from_line(self.matching_points[1], self.matching_points_count[1], 
                                                             self.matching_lines[0][0], self.matching_lines[0][1])
                        origo = get_closest_point(self.matching_lines[0], 2, point)
                        self.matching_lines[1][0] = origo
                        self.matching_lines[1][1] = point

                # Set angle tolerance for matching. Long lines have better precisions and therefore lower tolerance.
                for idx in range(2):
                    self.matching_angle_tol[idx] = max(math.radians(25 - 0.25 * line_length(self.matching_lines[idx])), math.radians(5))

                return

        # No good second feature was found, all points are probably in a straight line.
        #  We take the extreme points and put them in one feature each.
        self.matching_criterion       = MatchingCriterion.POINTS
        self.matching_points[0][0]    = end_points[0]
        self.matching_points_count[0] = 1
        self.matching_points[1][0]    = end_points[1]
        self.matching_points_count[1] = 1
        return
