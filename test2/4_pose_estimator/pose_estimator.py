
import cv2

class PoseEstimator:
    def __init__(self):
        pass

    def process(self, frames):

        gate_poses = []

        return gate_poses



if __name__ == '__main__':
    import pickle
    with open('../3_fiducial_matcher/frames.pickle', 'rb') as f:
        frames = pickle.load(f)

    print(frames)
    quit()

