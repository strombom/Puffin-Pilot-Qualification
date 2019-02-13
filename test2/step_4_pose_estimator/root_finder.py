

import sys
import pickle
import numpy as np

sys.path.insert(0, "../")
from common.gate_model import GateModel


class RootFinder:
    def __init__(self):

        camera_calibration = pickle.load(open("../common/camera_calibration/camera_calibration.pickle", "rb"))
        self.camera_matrix = camera_calibration['camera_matrix']
        self.dist_coefs = camera_calibration['dist_coefs']
        self.image_size = camera_calibration['image_size']

        self.gate_model = GateModel(self.camera_matrix, self.dist_coefs)


    def estimate_pose(self, points):



        points = np.array(points, np.float32)


        points = self.gate_model.undistort_points(points)

        print(points)
        quit()


        return None

