
import cv2
import math
import scipy
import numpy as np
from numba import jit, njit


class PointsFitter:
    def __init__(self, camera_matrix, dist_coefs, image_size, gate_model):
        self.gate_model = gate_model
        self.image_size = image_size

        map_x, map_y = cv2.initUndistortRectifyMap(camera_matrix, dist_coefs, None, camera_matrix, image_size, cv2.CV_32FC1)
        self.map_x = map_x
        self.map_y = map_y

    def fit(self, corners):
        # We need at least two corners
        if len(corners) < 2:
            return None

        if len(corners) == 2:
            # If we have only two corners, both of them must have lines that
            #  that indicate the perspective
            if not corners[0].has_lines or not corners[-1].has_lines:
                return None

        # Put the points into the correct sides, starting with the top left
        #  corner going clockwise
        corners = self._orient_corners(corners)
        points = [[], [], [], []]
        for i in range(len(corners)):
            if corners[i] is None:
                continue
            for side in range(2):
                side_idx = (i+side+3)%4
                for k in range(corners[i].points_count[side]):
                    point = corners[i].points[side][k]
                    points[side_idx].append(point)
        points_count = sum([len(i) for i in points])

        # Undistort the points
        points = self._undistorted_points(points)
       
        # Initial guess (rvec...tvec), camera placed very close to the gate pointing straight forward
        initial_guess = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        # Find a camera pose so that the distance to all points is minimized
        result = scipy.optimize.least_squares(fun       = self._points_loss, 
                                              x0        = initial_guess,
                                              args      = (points, points_count),
                                              ftol      = 1e-2,
                                              max_nfev  = 40)

        #quadrilateral = self.gate_model.get_undistorted_fiducial_corners(rvec, tvec)

        if result.success:
            rvec = result.x[0:3]
            tvec = result.x[3:6]
            return rvec, tvec
        else:
            return None

    def _points_loss(self, camera_pose, points, points_count):
        residuals = np.empty(points_count, np.float64)

        # Project gate model fiducial corners into image plane
        rvec, tvec = camera_pose[0:3], camera_pose[3:6]
        quadrilateral = self.gate_model.get_undistorted_fiducial_corners(rvec, tvec)

        point_idx = 0
        for i in range(4):
            for j in range(len(points[i])):
                # Calculate distance from point to side
                ls1 = quadrilateral[i]
                ls2 = quadrilateral[(i + 1) % 4]
                point = points[i][j]
                l2 = np.sum(np.square(ls2 - ls1))
                if l2 == 0:
                    distance = np.sum(np.square(point - ls1))
                else:
                    t = max(0.0, min(1.0, dot(point - ls1, ls2 - ls1) / l2))
                    proj = ls1 + t * (ls2 - ls1)
                    distance = np.sum(np.square(point - proj))
                residuals[point_idx] = distance
                point_idx += 1
        return residuals

    def _undistorted_points(self, points):
        # We use an undistortion map, its fast but we lose subpixel precision
        for side_idx in range(len(points)):
            if len(points[side_idx]) == 0:
                continue
            points[side_idx] = np.array(points[side_idx]).astype(np.int64)
            side = points[side_idx]
            side[:,0] = np.maximum(0, np.minimum(self.image_size[0] - 1, side[:,0]))
            side[:,1] = np.maximum(0, np.minimum(self.image_size[1] - 1, side[:,1]))
            for i in range(len(side)):
                x, y = side[i]
                side[i][0] = self.map_x[y][x]
                side[i][1] = self.map_y[y][x]
        return points

    def _orient_corners(self, corners):
        while len(corners) < 4:
            corners.append(None)
        
        # Estimate the angle of each of the four corners
        scores = []
        for i in range(4):
            score = 0
            for j in range(4):
                if corners[j] is None:
                    continue
                if corners[j].has_lines:
                    v0 = corners[j].lines[0][1] - corners[j].lines[0][0]
                    v1 = corners[j].lines[1][1] - corners[j].lines[1][0]
                    vc = v0 + v1
                    aref = math.radians(135 + 90 * j)
                    vref = np.array((math.cos(aref) - math.sin(aref),
                                     math.sin(aref) + math.cos(aref)))
                    m1 = math.sqrt(vc[0]*vc[0] + vc[1]*vc[1])
                    m2 = math.sqrt(vref[0]*vref[0] + vref[1]*vref[1])
                    angle = np.arccos(np.dot(vc, vref) / (m1 * m2))
                    score += abs(angle)
            scores.append(score)
            corners = [corners[-1]] + corners[0:3]

        # Rotate the frame until the  top left corner is first in the list
        angle_idx = np.argmin(scores)
        for i in range((angle_idx+2)%4):
            corners = [corners[-1]] + corners[0:3]
        return corners


@njit
def det(a, b):
    return a[0] * b[1] - a[1] * b[0]

@njit
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

@njit
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(line1[0], line1[1]), det(line2[0], line2[1]))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

