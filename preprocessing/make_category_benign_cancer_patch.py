import os
import shutil
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

for idx in range(1,51):
    if not os.path.exists('/media/disk/han/dataset/traindataset_2048/original_img/{0:03d}/'.format(idx)):
        os.mkdir('/media/disk/han/dataset/traindataset_2048/original_img/{0:03d}/'.format(idx))
    if not os.path.exists('/media/disk/han/dataset/traindataset_2048/label/{0:03d}/'.format(idx)):
        os.mkdir('/media/disk/han/dataset/traindataset_2048/label/{0:03d}/'.format(idx))

    vialbe_dir = glob.glob( '/media/disk/han/dataset/2048/viable_label/{0:03d}/*.png'.format(idx))
    whole_dir= glob.glob('/media/disk/han/dataset/2048/whole_label/{0:03d}/*.png'.format(idx))
    original_dir=glob.glob('/media/disk/han/dataset/2048/original_image/{0:03d}/*.png'.format(idx))

    for file in original_dir:
        name=file.split('/')
        vialbe_path='/media/disk/han/dataset/2048/viable_label/{0:03d}/'.format(idx)+name[-1]
        whole_path='/media/disk/han/dataset/2048/whole_label/{0:03d}/'.format(idx)+name[-1]


        original_img = cv2.imread(file)

          # original img 에서 배경 추출
        B, G, R = cv2.split(original_img)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

        background_label_img = B * G * R
        forground_label_img = np.ones((2048, 2048)) - background_label_img



        whole_img=cv2.imread(whole_path)


        if whole_img.sum()>409715:
            shutil.copy(file,'/media/disk/han/dataset/traindataset_2048/original_img/{0:03d}/'.format(idx)+name[-1])
            shutil.copy(vialbe_path,'/media/disk/han/dataset/traindataset_2048/label/{0:03d}/'.format(idx)+name[-1])