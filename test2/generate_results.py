#   ______                 ___       _______    
#  /_  __/__ ___ ___ _    / _ \__ __/ _/ _(_)__ 
#   / / / -_) _ `/  ' \  / ___/ // / _/ _/ / _ \
#  /_/  \__/\_,_/_/_/_/ /_/   \_,_/_//_//_/_//_/
# 
#
#              .-"-.
#             /  ,~a\_
#             \  \__))>
#             ,) ." \
#            /  (    \
#           /   )    ;
#          /   /     /
#        ,/_."`  _.-`
#         /_/`"\\___
#              `~~~`
#

import step_1_gate_finder             as gate_finder
import step_2_fiducial_finder         as fiducial_finder
import step_3_fiducial_matcher        as fiducial_matcher
import step_4_pose_estimator          as pose_estimator
import step_5_flying_region_generator as flying_region


class GenerateFinalDetections():
    def __init__(self, predict_dummy = True):
        self.gate_finder      = gate_finder.GateFinder()
        self.fiducial_finder  = fiducial_finder.FiducialFinder()
        self.fiducial_matcher = fiducial_matcher.FiducialMatcher()
        self.pose_estimator   = pose_estimator.PoseEstimator()
        self.flying_region    = flying_region.FlyingRegionGenerator()
        self.error_count = 0

        # Predict a dummy image to make sure that all jit methods
        # are compiled and that the tensorflow models are loaded.
        if predict_dummy:
            self.predict(None)

    def _predict_flying_regions(self, image, img_key):
        gate_image     = self.gate_finder.process(image)
        fiducials      = self.fiducial_finder.process(gate_image, img_key = img_key)
        frames         = self.fiducial_matcher.process(fiducials, gate_image = image, img_key = img_key)
        gate_poses     = self.pose_estimator.process(frames, gate_image = image, img_key = img_key)
        flying_regions = self.flying_region.process(gate_poses, gate_image = image, img_key = img_key)

        return flying_regions
        
    def predict(self, image, img_key = ""):
        print("predicting the new one", img_key)
        #try:
        return self._predict_flying_regions(image, img_key)
        #except:
        #    self.error_count += 1
        #    if self.error_count == 5:
        #        # Something is seriously wrong, reload everything.
        #        self.__init__(predict_dummy = False)
        #    return []
