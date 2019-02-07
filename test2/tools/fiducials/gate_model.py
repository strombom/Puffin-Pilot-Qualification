
import cv2
import numpy as np


class GateModel:
    gate_size = 11 * 0.3048  # Assumed 11 ft = 3.3528 m
    gate_post_width = 0.4525 # Optimized
    fiducial_size = 0.1143   # Assumed 3/8 ft = 0.1143 m

    def __init__(self, camera_matrix, dist_coefs, pose = (0, 0, 0)):
        self.camera_matrix = camera_matrix
        self.dist_coefs    = dist_coefs
        self.pose          = np.array(pose, dtype=np.float32)

        # Init gate features in 3D, with origo in the center of the gate
        self.fiducial_corners   = self._make_fiducials_corners()
        self.fiducial_points    = self._make_fiducials_points()
        self.inner_corners      = self._make_inner_corners()
        self.light_corners      = self._make_light_corners()
        self.back_frame_corners = self._make_back_frame_corners()

    def fiducials_from_corners(self, corners):
        # Sort fiducials, first top left, clockwise
        corners = corners[corners[:,0].argsort()]
        fid_left, fid_right = corners[0:2], corners[2:4]
        fid_left  = fid_left [fid_left [:,1].argsort()]
        fid_right = fid_right[fid_right[:,1].argsort()]
        corners[:] = [fid_left [0], fid_right[0], \
                           fid_right[1], fid_left [1]]
        fiducials_2d = np.ones((corners.shape[0], 1, 2))
        fiducials_2d[:,0,:] = corners

        # Estimate camera position
        success, rvec, tvec = cv2.solvePnP(self.fiducial_corners, 
                                           fiducials_2d, 
                                           self.camera_matrix, 
                                           self.dist_coefs)

        # Project all gate model fiducials on image
        fiducials, jac = cv2.projectPoints(self.fiducial_points, 
                                           rvec, 
                                           tvec, 
                                           self.camera_matrix, 
                                           self.dist_coefs)
        return fiducials[:,0,:]


    def _rotate_points(self, points, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s, 0), (s, c, 0), (0, 0, 0)))
        return points.dot(R)

    def _make_corners(self, p, z = 0.0):
        corners = np.empty((4, 1, 3))
        for i in range(4):
            corners[i][0] = self._rotate_points(np.array([(-p, -p, z)]), -i * np.radians(90))[0]
        return corners

    def _make_inner_corners(self):
        pass

    def _make_light_corners(self):
        pass
        
    def _make_back_frame_corners(self):
        pass
    
    def _make_fiducials_corners(self):
        p = self.gate_size / 2 - self.fiducial_size * 2
        return self._make_corners(p)

    def _make_fiducials_points(self):
        p = self.gate_size / 2 - self.fiducial_size * 2
        points, corner = np.empty((40, 3)), np.empty((10, 3))
        count = [5, 6]
        for ci in range(4):
            for i in range(count[0]):
                corner[i] = (-p, -p + (count[0] - i - 1) * self.fiducial_size, 0)
            for i in range(count[1] - 1):
                corner[i + count[0]] = (-p + (1 + i) * self.fiducial_size, -p, 0)
            points[ci*10:ci*10+10] = self._rotate_points(corner, -ci * np.radians(90))
            count = count[::-1]
        return points


