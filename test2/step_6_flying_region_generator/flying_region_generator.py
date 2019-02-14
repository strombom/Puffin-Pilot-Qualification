
import cv2
import sys
import pickle

sys.path.insert(0, "../")
from common.gate_model import GateModel


class FlyingRegionGenerator:
    def __init__(self):
        camera_calibration = pickle.load(open("../common/camera_calibration/camera_calibration.pickle", "rb"))
        self.camera_matrix = camera_calibration['camera_matrix']
        self.dist_coefs    = camera_calibration['dist_coefs']
        self.image_size    = camera_calibration['image_size']

        self.gate_model = GateModel(self.camera_matrix, self.dist_coefs)

    def process(self, gate_poses):
        flying_regions = []
        for n, gate_pose in enumerate(gate_poses):

            flying_region = self.get_flying_region(gate_pose)

            print(flying_region)
            quit()
            pass

        return flying_regions

    def get_flying_region(self, gate_pose):
        corners = self.gate_model.get_distorted_flying_region(gate_pose['rvec'], gate_pose['tvec'])
        inner_corners, light_corners, back_frame_corners = corners

        print(inner_corners)
        print(light_corners)
        print(back_frame_corners)

        

        quit()


if __name__ == '__main__':
    flying_region_generator = FlyingRegionGenerator()

    import pickle
    with open('../step_4_pose_estimator/gate_poses.pickle', 'rb') as f:
        gate_poses = pickle.load(f)

    flying_regions = flying_region_generator.process(gate_poses)

    for flying_region in flying_regions:
        print(flying_region)

    with open('flying_regions.pickle', 'wb') as f:
        pickle.dump(flying_regions, f)


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
