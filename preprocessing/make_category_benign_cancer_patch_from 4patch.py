import os
import shutil
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

for idx in range(1, 51):
    if not os.path.exists('/media/disk/han/dataset/Segmentation_whole/original_image/{0:03d}'.format(idx)):
        os.mkdir('/media/disk/han/dataset/Segmentation_whole/original_image/{0:03d}'.format(idx))
    if not os.path.exists('/media/disk/han/dataset/Segmentation_whole/whole_label_image/{0:03d}'.format(idx) ):
        os.mkdir('/media/disk/han/dataset/Segmentation_whole/whole_label_image/{0:03d}'.format(idx))
    label_path = '/media/disk/han/dataset/cancer_viable_label_patch/{0:03d}_whole_label/'.format(idx)

    pnglist = glob.glob(label_path + '/*.png')

    for file in pnglist:
        s = file.split('/')
        original_path = '/media/disk/han/dataset/original_patch/{0:03d}/'.format(idx) + s[-1]
        original_img = cv2.imread(original_path)
        whole_img = cv2.imread(file)

        B, G, R = cv2.split(original_img)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

        background_label_img = B * G * R
        forground_label_img = np.ones((1024, 1024)) - background_label_img

        if forground_label_img.sum() < 109715:
            continue

        shutil.copy(original_path,
                        '/media/disk/han/dataset/Segmentation_whole/original_image/{0:03d}'.format(idx)+ '/' + s[-1])

        shutil.copy(file,
                        '/media/disk/han/dataset/Segmentation_whole/whole_label_image/{0:03d}'.format(idx) + '/' + s[-1])
