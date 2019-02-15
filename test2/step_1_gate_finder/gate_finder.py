
import os
import cv2


class GateFinder:
    #
    # This class is supposed to estimate an approximate gate location(s) in
    #  the image and crop/resize the image to a standardized dimension.
    #
    # Currently the image is passed on without any processing.
    #

    def __init__(self):
        pass

    def process(self, image):
        if image is None:
            source_path = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(source_path, 'dummy_image.jpg')
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
