
import os
import cv2
import sys
import pickle

sys.path.insert(0, "../")
from step_3_fiducial_matcher import frames
from common.gate_model import GateModel

sys.modules['frames'] = frames

from .points_fitter import PointsFitter
from .quadrilateral_fitter import QuadrilateralFitter


class PoseEstimator:
    def __init__(self):
        source_path = os.path.dirname(os.path.abspath(__file__))
        calibration_path = os.path.join(source_path, "../common/camera_calibration/camera_calibration.pickle")
        camera_calibration = pickle.load(open(calibration_path, "rb"), encoding='bytes')
        self.camera_matrix = camera_calibration[b'camera_matrix']
        self.dist_coefs    = camera_calibration[b'dist_coefs']
        self.image_size    = camera_calibration[b'image_size']

        self.gate_model           = GateModel(self.camera_matrix, self.dist_coefs)
        self.points_fitter        = PointsFitter(self.camera_matrix, self.dist_coefs, self.image_size, self.gate_model)
        self.quadrilateral_fitter = QuadrilateralFitter(self.camera_matrix, self.dist_coefs, self.image_size, self.gate_model)

    def process(self, frames):
        if frames is None:
            return []

        gate_poses = []
        for n, frame in enumerate(frames):
            # First, try to use the quadrilateral fitter that uses the corners
            #  to estimate a quadrilateral directly on the undistorted image plane.
            result = self.quadrilateral_fitter.fit(frame.corners)

            # If the quadrilateral fitter fails, try to use the fiducial points
            #  to estimate the camera pose in 3D space. This is slower but can 
            #  handle missing corners.
            if result is None:
                result = self.points_fitter.fit(frame.corners)

            # Append the result if a gate pose was found was found
            if result is not None:
                gate_poses.append({'rvec': result[0], 'tvec': result[1]})
                
                # Only return the first pose, we know there is only one gate
                return gate_poses

        return gate_poses


if __name__ == '__main__':
    import os
    import pickle
    source_path = os.path.dirname(os.path.abspath(__file__))
    frames_path = os.path.join(source_path, '../step_3_fiducial_matcher/frames.pickle')
    with open(frames_path, 'rb') as f:
        frames = pickle.load(f)

    pose_estimator = PoseEstimator()
    gate_poses = pose_estimator.process(frames)

    for gate_pose in gate_poses:
        print(gate_pose)

    frames_path = os.path.join(source_path, 'gate_poses.pickle')
    with open('gate_poses.pickle', 'wb') as f:
        pickle.dump(gate_poses, f)



"""
        import cv2
        from seaborn import color_palette
        palette = color_palette("bright", 2)
        image_filepath = '../step_3_fiducial_matcher/img_in.png'
        image = cv2.imread(image_filepath)

        color = palette[0]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        for i in range(4):
            p0 = quadrilateral[i].astype(np.int64)
            p1 = quadrilateral[(i+1)%4].astype(np.int64)
            cv2.line(image, tuple(p0), tuple(p1), color, 2)

        color = palette[1]
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

        for point in points:
            point = point.astype(np.int64)
            cv2.circle(image, tuple(point), 3, color, -1)


        cv2.imwrite('img_quad.png', image)
        quit()
"""
