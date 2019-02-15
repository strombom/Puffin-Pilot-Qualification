
import numpy as np


class PillarFinder:
    def __init__(self):
        pass

    def process(self, gate_poses, gate_image):
        pillars = []
        return pillars

if __name__ == '__main__':
    pillar_finder = PillarFinder()

    import pickle
    with open('../step_4_pose_estimator/gate_poses.pickle', 'rb') as f:
        gate_poses = pickle.load(f)

    pillars = pose_estimator.process(None, gate_poses)

    for pillar in pillars:
        print(pillar)

    with open('pillars.pickle', 'wb') as f:
        pickle.dump(pillars, f)
