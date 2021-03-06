
import cv2
import math
import scipy
import numpy as np
from numba import jit, njit



class QuadrilateralFitter:
    def __init__(self, camera_matrix, dist_coeffs, image_size, gate_model):
        self.gate_model = gate_model
        self.image_size = image_size

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.gate_model = gate_model

    def fit(self, corners, img_key = ""):
        points = []
        initial_guess = np.empty((4, 2), np.float64)

        if len(corners) < 3:
            return None

        # Make sure that all corners have lines
        for i in range(len(corners)):
            if not corners[i].has_lines:
                return None

        if len(corners) == 3:
            # First and last sides of the frame must be lines to fit a quadrilateral
            if not corners[0].has_lines or not corners[-1].has_lines:
                return None

            # Find the missing corner point
            intersection = line_intersection(corners[0].lines[0], corners[-1].lines[1])
            if intersection[0] == -1:
                # No intersection
                return None

            # Put the estimated corner point as the last point in the initial guess
            initial_guess[3] = intersection
            points.append(intersection)

        for idx, corner in enumerate(corners):
            print("corner", idx)
            if corner.has_lines:
                initial_guess[idx] = corner.lines[0][0]
                points.append(corner.lines[0][0])
                points.append(corner.lines[0][1])
                points.append(corner.lines[1][1])
            else:
                initial_guess[idx] = corner.points[0][0]
                for i in range(2):
                    for j in range(corner.points_count[i]):
                        print(" add point", corner.points[i][j])
                        points.append(corner.points[i][j])
        points = np.array(points)

        # Undistort the points
        points = self._undistorted_points(points)

        # Find a quadrilateral so that the distance to all points is minimized
        result = scipy.optimize.least_squares(fun       = quadrilateral_loss, 
                                              x0        = np.reshape(initial_guess, (8,)),
                                              args      = (points,),
                                              ftol      = 1e-2,
                                              max_nfev  = 20)
        quadrilateral = np.reshape(result.x, (4, 2))

        # Estimate the camera position
        success, rvec, tvec = self.gate_model.camera_position_from_corners(quadrilateral, use_distortion = False)

        if success:
            return rvec, tvec, result.optimality
        else:
            return None

    def _undistorted_points(self, points):
        # Undistort points to 3D space
        points = np.reshape(points, (points.shape[0], 1, 2)).astype(np.float32)
        object_points = np.zeros((points.shape[0], 1, 3))
        object_points[:,:,0:2] = cv2.undistortPoints(points, self.camera_matrix, self.dist_coeffs)
        points, jac = cv2.projectPoints(objectPoints = object_points,
                                         rvec = np.zeros(3),
                                         tvec = np.zeros(3),
                                         cameraMatrix = self.camera_matrix,
                                         distCoeffs = None)
        points = points.reshape((points.shape[0], 2))
        return points

@njit
def quadrilateral_loss(quadrilateral, points):
    quadrilateral = np.reshape(quadrilateral, (4, 2))
    residuals = np.full(shape       = len(points),
                        fill_value = 1e9,
                        dtype      = np.float64)

    # Measure the shortest distances between the points and
    #  the four line segments of the quadrilateral.
    for i in range(len(points)):
        for j in range(4):
            ls1 = quadrilateral[j]
            ls2 = quadrilateral[(j + 1) % 4]
            point = points[i]
            l2 = np.sum(np.square(ls2 - ls1))
            if l2 == 0:
                distance = np.sum(np.square(point - ls1))
            else:
                t = max(0.0, min(1.0, dot(point - ls1, ls2 - ls1) / l2))
                proj = ls1 + t * (ls2 - ls1)
                distance = np.sum(np.square(point - proj))
            # Save the shortest distance for each point
            residuals[i] = min(residuals[i], distance)
    return residuals

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
    x =  det(d, xdiff) / div
    y =  det(d, ydiff) / div
    return x, y

"""
        from seaborn import color_palette
        palette = color_palette("bright", 3)
        image_filepath = '../step_1_gate_finder/dummy_image.jpg'
        image = cv2.imread(image_filepath)

        for idx in range(len(points)):
            color = palette[0]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
            point = points[idx].astype(np.int64)
            cv2.circle(image, tuple(point), 3, color, -1)

        cv2.imwrite('img_out.png', image)
"""
