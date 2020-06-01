import os
import shutil
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
from skimage.external import tifffile
for k in range(40,41):
    label_dir = 'D:/PAIP_cancer_label_patch/Training_phase_2_{0:03d}_viable_label'.format(k)
    for r, _, filenames in os.walk(label_dir):

        part_path = r.split('/')
        if not part_path[2] == '':
            part_dir = part_path[2].split('_')
            if part_dir[-2] == 'viable':
                original_whole_img_path = glob.glob('D:/PAIP_original_patch/' + part_path[2][:-13] + '/*.tif')

                tumor_img = tifffile.imread(original_whole_img_path[0])
                h, w = tumor_img.shape
                whole_label_img = np.zeros((h//1024, w//1024))
                viable_label_img = np.zeros((h//1024, w//1024))
                for filename in filenames:

                    viable_label_img_path = r + '/' + filename
                    whole_label_img_path = r[:-12] + 'whole_label/' + filename
                    original_img_path = 'D:/PAIP_original_patch/' + part_path[2][:-13] + '/' + filename

                    f_sp = filename.split('.')
                    h_idx, w_idx = f_sp[0].split('_')
                    h_idx = int(h_idx)
                    w_idx = int(w_idx)

                    original_img = cv2.imread(original_img_path)
                    viable_img = cv2.imread(viable_label_img_path)
                    whole_img = cv2.imread(whole_label_img_path)

                    # 답 차원 줄이기
                    _, viable_img = cv2.threshold(viable_img, 250, 1, cv2.THRESH_BINARY)
                    _, whole_img = cv2.threshold(whole_img, 250, 1, cv2.THRESH_BINARY)

                    # original img 에서 배경 추출
                    B, G, R = cv2.split(original_img)
                    _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
                    _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
                    _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

                    background_label_img = B * G * R
                    forground_label_img = np.ones((1024, 1024)) - background_label_img

                    if forground_label_img.sum() < 209715:
                        whole_label_img[h_idx, w_idx] = 0
                        viable_label_img[h_idx, w_idx] = 0
                        continue

                    if (forground_label_img * whole_img[:, :, 0]).sum() / forground_label_img.sum() >= 0.5:
                        whole_label_img[h_idx, w_idx] = 255

                        if (forground_label_img * viable_img[:, :, 0]).sum() / forground_label_img.sum() >= 0.5:
                            viable_label_img[h_idx, w_idx] = 255
                        else:
                            viable_label_img[h_idx, w_idx] = 0
                    else:
                        whole_label_img[h_idx, w_idx] = 0

                cv2.imwrite('label_patch/' + str(k) + '_tumor.jpg', whole_label_img)
                cv2.imwrite('label_patch/' + str(k) + '_viable.jpg', viable_label_img)
