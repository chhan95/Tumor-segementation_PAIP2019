import os
import shutil
import re
import cv2
import matplotlib.pyplot as plt
import numpy as np

for idx in range(21,51):
    label_dir = 'D:/PAIP_cancer_label_patch/Training_phase_2_{0:03d}_viable_label'.format(idx)
    for r, _, filenames in os.walk(label_dir):
        part_path = r.split('/')
        if not part_path[2] == '':
            part_dir = part_path[2].split('_')
            if part_dir[-2] == 'viable':
                for filename in filenames:
                    viable_label_img_path = r + '/' + filename
                    whole_label_img_path = r[:-12] + 'whole_label/' + filename
                    original_img_path = 'D:/PAIP_original_patch/' + part_path[2][:-13] + '/' + filename
                    original_img = cv2.imread(original_img_path)

                    whole_img = cv2.imread(whole_label_img_path)

                    # 답 차원 줄이기
                    _, whole_img = cv2.threshold(whole_img, 250, 1, cv2.THRESH_BINARY)

                    # original img 에서 배경 추출
                    B, G, R = cv2.split(original_img)
                    _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
                    _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
                    _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

                    background_label_img = B * G * R
                    forground_label_img = np.ones((1024, 1024)) - background_label_img

                    if forground_label_img.sum() < 209715:
                        continue
                    print(filename)
                    print((forground_label_img * whole_img[:, :, 0]).sum())
                    print(forground_label_img.sum())

                    if not os.path.exists('D:/PAIP_benign_cancer_patch/Cancer/' + part_path[2][:-13]):
                        os.mkdir('D:/PAIP_benign_cancer_patch/Cancer/' + part_path[2][:-13])
                    if not os.path.exists('D:/PAIP_benign_cancer_patch/Benign/' + part_path[2][:-13]):
                        os.mkdir('D:/PAIP_benign_cancer_patch/Benign/' + part_path[2][:-13])
                    if not os.path.exists('D:/PAIP_benign_cancer_patch/Ambiguity/' + part_path[2][:-13]):
                        os.mkdir('D:/PAIP_benign_cancer_patch/Ambiguity/' + part_path[2][:-13])

                    if (forground_label_img * whole_img[:, :, 0]).sum() / forground_label_img.sum() > 0.8:
                        shutil.copy(original_img_path,
                                    'D:/PAIP_benign_cancer_patch/Cancer/' + part_path[2][:-13] + '/' + filename)
                    elif (forground_label_img * whole_img[:, :, 0]).sum() / forground_label_img.sum() < 0.2:
                        shutil.copy(original_img_path,
                                    'D:/PAIP_benign_cancer_patch/Benign/' + part_path[2][:-13] + '/' + filename)
                    else:
                        shutil.copy(original_img_path,
                                    'D:/PAIP_benign_cancer_patch/Ambiguity/' + part_path[2][:-13] + '/' + filename)

