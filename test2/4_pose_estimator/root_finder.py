

import sys
import pickle

sys.path.insert(0, "../")
from common.gate_model import GateModel


class RootFinder:
    def __init__(self):

        self.camera_calibration = pickle.load(open("../common/camera_calibration/camera_calibration.pickle", "rb"))

    def estimate_pose(self, points):

        return None

