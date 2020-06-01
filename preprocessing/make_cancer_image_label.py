from skimage.external import tifffile
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil



original_target_path = '/media/disk/han/dataset/for_segmentation/original_patch'
whole_target_path = '/media/disk/han/dataset/for_segmentation/cancer_viable_label_patch'

original_path = '/media/disk/han/dataset/original_patch/'
label_path = '/media/disk/han/dataset/cancer_viable_label_patch/'

for idx in range(1, 51):
    print(idx)

    if not os.path.exists(
            original_target_path + '/{0:03d}'.format(idx)):
        os.mkdir(original_target_path + '/{0:03d}'.format(idx))
    if not os.path.exists(
            whole_target_path + '/{0:03d}'.format(idx)):
        os.mkdir(
            whole_target_path + '/{0:03d}'.format(idx))

    file_path = label_path + '{0:03d}_viable_label/'.format( idx)
    file_list = glob.glob(file_path + '*.png')

    for file_name in file_list:
        label_img = cv2.imread(file_name)
        file=file_name.split('/')
        _, label_img = cv2.threshold(label_img, 250, 1, cv2.THRESH_BINARY)


        original_img=cv2.imread(original_path+'/{0:03d}/'.format( idx)+file[-1])
        R, G, B = cv2.split(original_img)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)

        background_label_img = R * B * G
        forground_label_img = np.ones((1024, 1024)) - background_label_img

        if forground_label_img.sum()>20000:
            shutil.copy(file_name,
                        whole_target_path + '/{0:03d}/'.format(idx)+file[-1])
            shutil.copy(original_path+'/{0:03d}/'.format( idx)+file[-1],
                        original_target_path + '/{0:03d}/'.format(idx) + file[-1])
