
import os
import cv2
import sys
import pickle
import numpy as np
from numba import njit

sys.path.insert(0, "../")
from common.gate_model import GateModel

import random

import cv2
from seaborn import color_palette


class FlyingRegionGenerator:
    def __init__(self):
        source_path = os.path.dirname(os.path.abspath(__file__))
        calibration_path = os.path.join(source_path, "../common/camera_calibration/camera_calibration.pickle")
        camera_calibration = pickle.load(open(calibration_path, "rb"))
        self.camera_matrix = camera_calibration['camera_matrix']
        self.dist_coefs    = camera_calibration['dist_coefs']
        self.image_size    = camera_calibration['image_size']

        self.gate_model = GateModel(self.camera_matrix, self.dist_coefs)

    def process(self, gate_poses, pillars = None, gate_image = None, img_key = ""):
        flying_regions = []
        for n, gate_pose in enumerate(gate_poses):
            flying_region, points = self.get_flying_region(gate_pose)
            flying_regions.append(flying_region.tolist())

            if gate_image is not None:

                palette = color_palette("bright", 4)
                #if gate_image is not None:
                image = cv2.cvtColor(gate_image, cv2.COLOR_RGB2BGR)
                #else:
                #    image_filepath = '../step_1_gate_finder/dummy_image.jpg'
                #    image = cv2.imread(image_filepath)

                color = palette[0]
                color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

                polygon = np.rint(flying_region).astype(np.integer)

                cv2.fillConvexPoly(image, polygon, (color))


                for point in points:
                    point = point.astype(np.int64)
                    cv2.circle(image, tuple(point), 2, color, -1)


                """
                for i, corner in enumerate(corners):
                    color = palette[i]
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    
                    for j in range(2):
                        for k in range(corner.points_count[j]):
                            point = corner.points[j][k].astype(np.int64)
                            #cv2.circle(image, tuple(point), 3, color, -1)

                #cv2.imwrite('img_out.png', image)



                inner_corners, light_corners, back_frame_corners = self.gate_model.get_distorted_flying_region(rvec, tvec)


                polygon = np.rint(flying_region).astype(np.integer)
                cv2.fillConvexPoly(image, polygon, (color))

                for i, corner in enumerate(inner_corners):
                    color = palette[1]
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    point = corner.astype(np.int64)
                    cv2.circle(image, tuple(point), 3, color, -1)

                for i, corner in enumerate(light_corners):
                    color = palette[2]
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    point = corner.astype(np.int64)
                    cv2.circle(image, tuple(point), 3, color, -1)

                for i, corner in enumerate(back_frame_corners):
                    color = palette[3]
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    point = corner.astype(np.int64)
                    cv2.circle(image, tuple(point), 3, color, -1)
                """
                filename = 'img_out_%s.png' % img_key
                print("filename", filename)
                cv2.imwrite(filename, image)


        return flying_regions



        #import cv2
        #from seaborn import color_palette

        palette = color_palette("bright", len(flying_regions))
        #image_filepath = '../step_3_fiducial_matcher/img_in.png'
        source_path = os.path.dirname(os.path.abspath(__file__))
        image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
        image = cv2.imread(image_filepath)

        for i in range(len(flying_regions)):

            color = palette[i]
            color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))

            flying_region, points = flying_regions[i]

            polygon = np.rint(flying_region).astype(np.integer)

            cv2.fillConvexPoly(image, polygon, (color))

            for point in points:
                point = point.astype(np.int64)
                cv2.circle(image, tuple(point), 2, color, -1)

        print(flying_regions)
        print("save")
        image_filepath = os.path.join(source_path, 'img_region.png')
        cv2.imwrite(image_filepath, image)
        quit()


        return flying_regions

    def get_flying_region(self, gate_pose):
        corners = self.gate_model.get_distorted_flying_region(gate_pose['rvec'], gate_pose['tvec'])
        inner_corners, light_corners, back_frame_corners = corners

        corners = np.empty((4, 3, 2))
        corners[:,0,:] = inner_corners
        corners[:,1,:] = light_corners
        corners[:,2,:] = back_frame_corners

        cog = self._get_center_of_gravity(corners)

        points = []

        quadrilatreal = np.empty((4, 2))
        for c_idx in range(4):
            c0 = corners[c_idx]
            c1 = corners[(c_idx+1)%4]
            c2 = corners[(c_idx+3)%4]

            min_corner = np.empty(2)
            min_distance = 1e9
            for i in range(3):
                for j in range(3):
                    x10, y10 = c0[j][0], c0[j][1]
                    x20, y20 = c0[i][0], c0[i][1]

                    x11, y11 = c1[j][0], c1[j][1]
                    x21, y21 = c2[i][0], c2[i][1]

                    l1 = np.array(((x10, y10), (x11, y11)))
                    l2 = np.array(((x20, y20), (x21, y21)))

                    point = line_intersection(l1, l2)

                    points.append(np.array(point))

                    v = point - cog
                    distance = v[0]*v[0] + v[1] * v[1]
                    if distance < min_distance:
                        min_distance = distance
                        min_corner = point

            quadrilatreal[c_idx] = min_corner

        return quadrilatreal, points

    def _get_center_of_gravity(self, corners):
        cog_x = np.sum(corners[:,:,0].reshape((12))) / 12.0
        cog_y = np.sum(corners[:,:,1].reshape((12))) / 12.0
        cog = np.array((cog_x, cog_y))
        return cog

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



if __name__ == '__main__':
    import os
    import pickle
    source_path = os.path.dirname(os.path.abspath(__file__))
    gate_poses_path = os.path.join(source_path, '../step_4_pose_estimator/gate_poses.pickle')
    with open(gate_poses_path, 'rb') as f:
        gate_poses = pickle.load(f)

    source_path = os.path.dirname(os.path.abspath(__file__))
    image_filepath = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
    image = cv2.imread(image_filepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    flying_region_generator = FlyingRegionGenerator()
    flying_regions = flying_region_generator.process(gate_poses, pillars = None, gate_image = image, img_key = "")

    for flying_region in flying_regions:
        print(flying_region)

    flying_regions_path = os.path.join(source_path, 'flying_regions.pickle')
    with open(flying_regions_path, 'wb') as f:
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



        import cv2
        from seaborn import color_palette
        palette = color_palette("bright", 2)
        source_path = os.path.dirname(os.path.abspath(__file__))
        image_filepath = os.path.join(source_path, '../step_2_fiducial_finder/fiducials.png')
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
