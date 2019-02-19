
import cv2
import math
import scipy
import numpy as np
from numba import jit, njit


class PointsFitter:
    def __init__(self, camera_matrix, dist_coeffs, image_size, gate_model):
        self.gate_model = gate_model
        self.image_size = image_size

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.gate_model = gate_model

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
        corner_points = [[], [], [], [], []]
        for i in range(len(corners)):
            if corners[i] is None:
                continue
            if corners[i].has_lines:
                for side in range(2):
                    side_idx = (i+side+3)%4
                    for k in range(corners[i].points_count[side]):
                        point = corners[i].points[side][k]
                        corner_points[side_idx].append(point)
            else:
                for side in range(2):
                    for k in range(corners[i].points_count[side]):
                        point = corners[i].points[side][k]
                        corner_points[4].append(point)
                continue

        corner_points_count = sum([len(i) for i in corner_points])

        # Undistort the points
        points = self._undistorted_points(corner_points)
       
        # Initial guess (rvec...tvec), camera placed very close to the gate pointing straight forward
        initial_guess = np.array((0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        # Find a camera pose so that the distance to all points is minimized
        result = scipy.optimize.least_squares(fun       = self._points_loss, 
                                              x0        = initial_guess,
                                              args      = (corner_points, corner_points_count),
                                              ftol      = 1e-2,
                                              max_nfev  = 30)

        if result.success:
            rvec = result.x[0:3]
            tvec = result.x[3:6]
            return rvec, tvec
        else:
            return None

    def _points_loss(self, camera_pose, points, points_count):
        residuals = np.full(points_count, 1e9, np.float64)

        # Project gate model fiducial corners into image plane
        rvec, tvec = camera_pose[0:3], camera_pose[3:6]
        quadrilateral = self.gate_model.get_undistorted_fiducial_corners(rvec, tvec)

        jit_points_loss(camera_pose, points[0], points[1], points[2], points[3], points[4], residuals, rvec, tvec, quadrilateral)

        return residuals

    def _undistorted_points(self, corners):
        # Undistort points to 3D space
        for idx in range(len(corners)):
            points = np.array(corners[idx])
            if len(points) > 0:
                points = np.reshape(points, (points.shape[0], 1, 2)).astype(np.float32)
                object_points = np.zeros((points.shape[0], 1, 3))
                object_points[:,:,0:2] = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs)
                points, jac = cv2.projectPoints(objectPoints = object_points,
                                                 rvec = np.zeros(3),
                                                 tvec = np.zeros(3),
                                                 cameraMatrix = self.camera_matrix,
                                                 distCoeffs = None)
                if points.shape[0] == 0:
                    points = np.empty((0, 2))
                else:
                    points = points.reshape((points.shape[0], 2))
            else:
                points = np.empty((0, 2))
            corners[idx] = points
        return corners

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

        # Rotate the frame until the top left corner is first in the list
        angle_idx = np.argmin(scores)
        for i in range((angle_idx+2)%4):
            corners = [corners[-1]] + corners[0:3]
        return corners

@njit
def jit_points_loss(camera_pose, points0, points1, points2, points3, points4, residuals, rvec, tvec, quadrilateral):
    points = [points0, points1, points2, points3, points4]

    # Loss for points where there is a known line segment
    point_idx = 0
    for quad_i in range(4):
        for point_j in range(points[quad_i].shape[0]):
            # Calculate distance from point to side
            ls1 = quadrilateral[(quad_i) % 4]
            ls2 = quadrilateral[(quad_i + 1) % 4]
            point = points[quad_i][point_j]
            l2 = np.sum(np.square(ls2 - ls1))
            if l2 == 0:
                v = point - ls1
                distance = v[0]*v[0] + v[1]*v[1]
            else:
                t = max(0.0, min(1.0, dot(point - ls1, ls2 - ls1) / l2))
                proj = ls1 + t * (ls2 - ls1)
                distance = np.sum(np.square(point - proj))**0.5
            residuals[point_idx] = min(residuals[point_idx], distance)
            point_idx += 1

    # Loss for remaining unknown points that still might help
    quad_i = 4
    for point_j in range(points[quad_i].shape[0]):
        for d in range(4):
            # Calculate distance from point to side
            ls1 = quadrilateral[(quad_i + d) % 4]
            ls2 = quadrilateral[(quad_i + d + 1) % 4]
            point = points[quad_i][point_j]
            l2 = np.sum(np.square(ls2 - ls1))
            if l2 == 0:
                v = point - ls1
                distance = v[0]*v[0] + v[1]*v[1]
            else:
                t = max(0.0, min(1.0, dot(point - ls1, ls2 - ls1) / l2))
                proj = ls1 + t * (ls2 - ls1)
                distance = np.sum(np.square(point - proj))**0.5
            residuals[point_idx] = min(residuals[point_idx], distance)
        point_idx += 1

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

