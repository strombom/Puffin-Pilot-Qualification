
import os
import random
from shutil import copyfile

original_path = '../../../data-puffin-pilot/Undistorted'
random_path = '../../../data-puffin-pilot/Random'


filenames = ['IMG_3293', 'IMG_7411', 'IMG_2766', 'IMG_0938', 'IMG_3963', 'IMG_4734', 'IMG_5001', 'IMG_1656', 'IMG_3376', 'IMG_9199', 'IMG_5393', 'IMG_5344', 'IMG_4553', 'IMG_6247', 'IMG_7448', 'IMG_6599', 'IMG_7255', 'IMG_4049', 'IMG_3869', 'IMG_4597', 'IMG_0320', 'IMG_9094', 'IMG_2809', 'IMG_5143', 'IMG_5418', 'IMG_3668', 'IMG_0975', 'IMG_2638', 'IMG_8131', 'IMG_8098', 'IMG_0406', 'IMG_0107', 'IMG_8007', 'IMG_3565', 'IMG_4851', 'IMG_1443', 'IMG_7738', 'IMG_4518', 'IMG_0905', 'IMG_8126', 'IMG_6600', 'IMG_1132', 'IMG_8220', 'IMG_4633', 'IMG_5781', 'IMG_2123', 'IMG_1564', 'IMG_8284', 'IMG_3367', 'IMG_2260', 'IMG_5957', 'IMG_3970', 'IMG_4656', 'IMG_4494', 'IMG_4363', 'IMG_0984', 'IMG_1539', 'IMG_5465', 'IMG_3494', 'IMG_6801', 'IMG_0861', 'IMG_8596', 'IMG_2431', 'IMG_6486', 'IMG_8018', 'IMG_5969', 'IMG_8044', 'IMG_2255', 'IMG_5844', 'IMG_8942', 'IMG_3672', 'IMG_2891', 'IMG_3147', 'IMG_2472', 'IMG_1210', 'IMG_0498', 'IMG_0670', 'IMG_3374', 'IMG_6737', 'IMG_5609', 'IMG_8003', 'IMG_5700', 'IMG_0913', 'IMG_3431', 'IMG_6471', 'IMG_0105', 'IMG_0844', 'IMG_8689', 'IMG_3821', 'IMG_6856', 'IMG_2351', 'IMG_2471', 'IMG_8979', 'IMG_8690', 'IMG_4806', 'IMG_5014', 'IMG_5454', 'IMG_7433', 'IMG_4035', 'IMG_6250.JPG']



numer_of_files = 100

for filename in random.sample(os.listdir(original_path), numer_of_files):
    print(filename)
    copyfile(os.path.join(original_path, filename), 
             os.path.join(random_path, filename))


