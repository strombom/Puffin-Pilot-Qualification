
import cv2
import scipy
import numpy as np

from gate_model_utils import quad_to_poly, make_corners, rotate_points, sort_corners


class GateModel:
    gate_size       = 11 * 0.3048 # Assumed 11 ft = 3.3528 m
    gate_post_width = 0.4525      # Optimized ~1.5 ft
    fiducial_size   = 0.1143      # Assumed 3/8 ft

    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix
        self.dist_coeffs    = dist_coeffs
        self.camera_pos    = np.zeros(3)
        self.camera_rot    = np.identity(3)

        # Init gate features in 3D, with origo in the center of the gate
        self.fiducial_corners   = self._make_fiducials_corners()
        self.fiducial_points    = self._make_fiducials_points()
        self.outer_corners      = self._make_outer_corners()
        self.inner_corners      = self._make_inner_corners()
        self.light_corners      = self._make_light_corners()
        self.back_frame_corners = self._make_back_frame_corners()

    def camera_position_from_corners(self, corners, use_distortion = True):
        # Sort fiducials, first top left, clockwise
        corners = sort_corners(corners)
        fiducials_2d = np.ones((corners.shape[0], 1, 2))
        fiducials_2d[:,0,:] = corners

        dist_coeffs = None
        if use_distortion:
            dist_coeffs = self.dist_coeffs

        # Estimate camera position
        success, rvec, tvec = cv2.solvePnP(objectPoints = self.fiducial_corners, 
                                           imagePoints = fiducials_2d, 
                                           cameraMatrix = self.camera_matrix, 
                                           distCoeffs = dist_coeffs)

        return success, rvec, tvec

    def fiducials_from_corners(self, corners):
        success, rvec, tvec = self.camera_position_from_corners(corners)

        if not success:
            return None

        # Project all gate model fiducials on image
        fiducials = self._get_distorted_points(rvec, tvec, self.fiducial_points)
        return fiducials[:,0,:]

    def get_undistorted_fiducial_corners(self, rvec, tvec):
        return self._get_undistorted_points(rvec, tvec, self.fiducial_corners)

    def get_distorted_front_polygons(self, fiducials):
        # Undistort points to 3D space
        fiducials = np.reshape(fiducials, (fiducials.shape[0], 1, 2)).astype(np.float32)
        object_points = cv2.undistortPoints(fiducials, self.camera_matrix, self.dist_coeffs)

        # Fit quadrilateral to fiducials
        quadrilateral = np.ones((4, 1, 2))
        quadrilateral[:,0,0:2] = self.fit_fiducials(object_points[:,0,:])

        # Estimate camera position
        success, rvec, tvec = cv2.solvePnP(objectPoints = self.fiducial_corners, 
                                           imagePoints = quadrilateral, 
                                           cameraMatrix = np.identity(3), 
                                           distCoeffs = None)

        # Project outer gate on image
        outer_polygon = self._get_distorted_points(rvec, tvec, quad_to_poly(self.outer_corners))

        # Project inner gate on image
        inner_polygon = self._get_distorted_points(rvec, tvec, quad_to_poly(self.inner_corners))

        return outer_polygon, inner_polygon

    def get_distorted_flying_region(self, rvec, tvec):
        inner_corners      = self._get_distorted_points(rvec, tvec, self.inner_corners)
        light_corners      = self._get_distorted_points(rvec, tvec, self.light_corners)
        back_frame_corners = self._get_distorted_points(rvec, tvec, self.back_frame_corners)

        return inner_corners, light_corners, back_frame_corners

    def _get_distorted_points(self, rvec, tvec, points):
        points, jac = cv2.projectPoints(objectPoints = points,
                                        rvec = rvec,
                                        tvec = tvec,
                                        cameraMatrix = self.camera_matrix,
                                        distCoeffs = self.dist_coeffs)
        return points.reshape((points.shape[0], 2))

    def _get_undistorted_points(self, rvec, tvec, points):
        points, jac = cv2.projectPoints(objectPoints = points,
                                        rvec = rvec,
                                        tvec = tvec,
                                        cameraMatrix = self.camera_matrix,
                                        distCoeffs = None)
        return points.reshape((points.shape[0], 2))

    def _make_outer_corners(self):
        p = self.gate_size / 2
        return make_corners(p)

    def _make_inner_corners(self):
        p = self.gate_size / 2 - self.gate_post_width
        return make_corners(p)

    def _make_light_corners(self):
        p = self.gate_size / 2 - self.gate_post_width - 0.04
        return make_corners(p, z = 0.15) + np.array((0, 0.01, 0))
        
    def _make_back_frame_corners(self):
        p = self.gate_size / 2 - self.gate_post_width + 0.02
        return make_corners(p, z = 0.35)
    
    def _make_fiducials_corners(self):
        p = self.gate_size / 2 - self.fiducial_size * 2
        return make_corners(p)

    def _make_fiducials_points(self):
        p = self.gate_size / 2 - self.fiducial_size * 2
        points, corner = np.empty((40, 3)), np.empty((10, 3))
        count = [5, 6]
        for ci in range(4):
            for i in range(count[0]):
                corner[i] = (-p, -p + (count[0] - i - 1) * self.fiducial_size, 0)
            for i in range(count[1] - 1):
                corner[i + count[0]] = (-p + (1 + i) * self.fiducial_size, -p, 0)
            points[ci*10:ci*10+10] = rotate_points(corner, -ci * np.radians(90))
            count = count[::-1]
        return points


