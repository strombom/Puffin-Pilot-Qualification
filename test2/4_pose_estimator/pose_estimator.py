
import cv2

from root_finder import RootFinder

class PoseEstimator:
    def __init__(self):
        self.root_finder = RootFinder()

    def process(self, frames):

        gate_poses = []

        for frame in frames:
            gate_pose = self.root_finder.estimate_pose(frame)
            gate_poses.append(gate_pose)

        return gate_poses


if __name__ == '__main__':
    pose_estimator = PoseEstimator()

    import pickle
    with open('../3_fiducial_matcher/frames.pickle', 'rb') as f:
        frames = pickle.load(f)

    gate_poses = pose_estimator.process(frames)

    print("end")
    print(gate_poses)
    quit()

