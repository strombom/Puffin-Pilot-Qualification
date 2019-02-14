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

from step_1_gate_finder               import GateFinder
from step_2_fiducial_finder           import FiducialFinder
from step_3_fiducial_matcher          import FiducialMatcher
from step_4_pose_estimator            import PoseEstimator
from step_5_pillar_finder             import PillarFinder
from step_6_flying_region_generator   import FlyingRegionGenerator


class GenerateFinalDetections():
    def __init__(self):
        self.gate_finder             = GateFinder()
        self.fiducial_finder         = FiducialFinder()
        self.fiducial_matcher        = FiducialMatcher()
        self.pose_estimator          = PoseEstimator()
        self.pillar_finder           = PillarFinder()
        self.flying_region_generator = FlyingRegionGenerator()

        # Predict a dummy image to make sure that all jit methods
        # are compiled and that the tensorflow models are loaded.
        self.predict(None)
        self.error_count = 0

    def _predict_flying_regions(self, image):
        gate_image     = self.gate_finder.process(image)
        fiducials      = self.fiducial_finder.process(gate_image)
        frames         = self.fiducial_matcher.process(fiducials)
        gate_poses     = self.pose_estimator.process(frames)
        pillars        = self.pillar_finder.process(gate_image, gate_poses)
        flying_regions = self.flying_region_generator.process(gates, pillars)

        return flying_regions
        
    def predict(self, image):
        try:
            return self._predict_flying_regions(image)
        except:
            self.error_count += 1
            if self.error_count == 5:
                # Something is seriously wrong, reload everything.
                self.__init__()
            return []
