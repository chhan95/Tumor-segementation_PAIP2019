import os
import shutil
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

for idx in range(1, 21):
    label_path = 'D:/new_4_label/Training_phase_1_{0:03d}_whole_label'.format(idx)

    jpglist = glob.glob(label_path + '/*.jpg')
    pnglist = glob.glob(label_path + '/*.png')
    i = 0
    for file in jpglist:
        s = file.split('\\')
        original_path = 'D:/new_4/Training_phase_1_{0:03d}/'.format(idx) + s[-1]

        original_img = cv2.imread(original_path)
        whole_img = cv2.imread(file)
        print('{0} {1} {2}'.format(i, s[-1], original_path))
        i += 1

        # original img 에서 배경 추출
        B, G, R = cv2.split(original_img)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

        background_label_img = B * G * R
        forground_label_img = np.ones((1024, 1024)) - background_label_img
        # 답 차원 줄이기
        _, whole_img = cv2.threshold(whole_img, 250, 1, cv2.THRESH_BINARY)

        if not os.path.exists('D:/4_PAIP_benign_cancer_patch/Cancer/Training_phase_1_{0:03d}'.format(idx)):
            os.mkdir('D:/4_PAIP_benign_cancer_patch/Cancer/Training_phase_1_{0:03d}'.format(idx))
        if not os.path.exists('D:/4_PAIP_benign_cancer_patch/Benign/Training_phase_1_{0:03d}'.format(idx)):
            os.mkdir('D:/4_PAIP_benign_cancer_patch/Benign/Training_phase_1_{0:03d}'.format(idx))

        if forground_label_img.sum() < 209715:
            continue
        if (forground_label_img * whole_img[:, :, 0]).sum() / forground_label_img.sum() >= 0.5:
            shutil.copy(original_path,
                        'D:/4_PAIP_benign_cancer_patch/Cancer/Training_phase_1_{0:03d}'.format(idx) + '/' + s[-1])
        else:
            shutil.copy(original_path,
                        'D:/4_PAIP_benign_cancer_patch/Benign/Training_phase_1_{0:03d}'.format(idx) + '/' + s[-1])
