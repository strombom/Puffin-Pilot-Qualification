
import os
import cv2
import numpy as np

from step_2_fiducial_finder.gatenet import gatenet

class FiducialFinder:
    def __init__(self):
        source_path = os.path.dirname(os.path.abspath(__file__))
        #weights_path = os.path.join(source_path, 'logdir/checkpoint_20190207_loss295')
        weights_path = os.path.join(source_path, 'logdir/checkpoint_20190214_loss264')

        self.model = gatenet()
        self.model.load_weights(weights_path)


    def process(self, image):
        #source_path = os.path.dirname(os.path.abspath(__file__))
        #filepath = os.path.join(source_path, "o" + '_mask.png')
        #cv2.imwrite(filepath, image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.reshape((1, image.shape[0], image.shape[1], 1))
        predictions = self.model.predict([image])

        #prediction = (predictions[0] * 255).astype(np.uint8)
        #filepath = os.path.join(source_path, "p" + '_mask.png')
        #cv2.imwrite(filepath, prediction)

        prediction = predictions.reshape((predictions.shape[1], predictions.shape[2]))
        
        #print(prediction)
        #filepath = os.path.join(source_path, "p3" + '_mask.png')
        #cv2.imwrite(filepath, (prediction * 255).astype(np.uint8))
        #quit()

        return prediction


if __name__ == '__main__':
    source_path = os.path.dirname(os.path.abspath(__file__))

    fiducial_finder = FiducialFinder()

    image_path = os.path.join(source_path, '../step_1_gate_finder/dummy_image.jpg')
    #image_path = os.path.join(source_path, '../step_1_gate_finder/IMG_2984.JPG')
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fiducials = fiducial_finder.process(image)

    filepath = os.path.join(source_path, 'fiducials.png')
    cv2.imwrite(filepath, (fiducials * 255).astype(np.uint8))

    import pickle
    fiducials_path = os.path.join(source_path, 'fiducials.pickle')
    with open(fiducials_path, 'wb') as f:
        pickle.dump(fiducials, f)

