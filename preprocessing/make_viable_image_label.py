from skimage.external import tifffile
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil

training_phase = 2

original_target_path = 'D:/Segmentation_viable/original_image'
viable_target_path = 'D:/Segmentation_viable/viable_label_image'

original_path = 'D:/PAIP_original_patch'
label_path = 'D:/PAIP_cancer_label_patch/'

for idx in range(21, 51):

    if not os.path.exists(
            original_target_path + '/Training_phase_{0}_{1:03d}'.format(training_phase, idx)):
        os.mkdir(original_target_path + '/Training_phase_{0}_{1:03d}'.format(training_phase, idx))
    if not os.path.exists(
            viable_target_path + '/Training_phase_{0}_{1:03d}'.format(training_phase, idx)):
        os.mkdir(
            viable_target_path + '/Training_phase_{0}_{1:03d}'.format(training_phase, idx))

    file_path = label_path + 'Training_phase_{0}_{1:03d}_viable_label/'.format(training_phase, idx)
    file_list = glob.glob(file_path + '*.jpg')

    for file_name in file_list:
        label_img = cv2.imread(file_name)
        file=file_name.split('\\')
        _, label_img = cv2.threshold(label_img, 250, 1, cv2.THRESH_BINARY)

        if label_img.sum() > 1000:
            shutil.copy(file_name,
                        viable_target_path + '/Training_phase_{0}_{1:03d}/'.format(training_phase, idx)+file[-1])
            shutil.copy(original_path+'/Training_phase_{0}_{1:03d}/'.format(training_phase, idx)+file[-1],
                        original_target_path + '/Training_phase_{0}_{1:03d}/'.format(training_phase, idx) + file[-1])