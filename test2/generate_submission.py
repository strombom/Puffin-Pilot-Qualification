# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from generate_results import *
import time

data_path = '/home/jst/development/data-puffin-pilot/Data_LeaderboardTesting_Difficult/'
#data_path = '/home/jst/development/data-puffin-pilot/Data_LeaderboardTesting/'
#data_path = 'testing/images/'

img_file = glob.glob(data_path + '*.JPG')
img_keys = [img_i.split('/')[-1] for img_i in img_file]

bad_keys = ["0365", 
            "0714", 
            "0715", 
            "0717", 
            "0926", 
            "1177", 
            "1437", 
            "1751", 
            "1915", 
            "2930", 
            "2969", 
            "3308", 
            "3319", 
            "3320", 
            "4741", 
            "4753", 
            "5368", 
            "5383", 
            "6829 (1)", 
            "6835 (1)", 
            "7512 (1)", 
            "7514 (1)", 
            "7522 (1)", 
            "7567 (1)", #<
            "7569 (1)", #<
            "7747 (1)", 
            "8556", 
            "8639", 
            "8649", 
            "8667", 
            "8668", 
            "8672", 
            "8692", 
            "8728 (1)", 
            "8753 (1)", 
            "8773", 
            "8994", 
            "9000", 
            "9003", 
            "9016", ]

# Instantiate a new detector
finalDetector = GenerateFinalDetections(predict_dummy = False)
# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for img_key in img_keys:
    #if img_key.replace("IMG_", "").replace(".JPG", "") not in bad_keys:
    #    continue
    img =cv2.imread(data_path+img_key)
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tic = time.monotonic()
    bb_all = finalDetector.predict(img, img_key)
    toc = time.monotonic()
    pred_dict[img_key] = bb_all
    time_all.append(toc-tic)

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

with open('submission_20190307.json', 'w') as f:
    json.dump(pred_dict, f)
