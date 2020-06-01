import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.external import tifffile
import os

import matplotlib.pyplot as plt


label_path = 'D:/WSI_label/'
prediction_path = 'D:/WSI_prediction/'
tumor_abs_path = 'D:/4_PAIP_cancer_label_patch/'
for i in range(1, 21):

    if not os.path.exists('D:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('D:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('D:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('D:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(i)):
        os.mkdir('D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(i))

    tumor_label_path = label_path + str(i) + '_tumor.jpg'
    viable_label_path = label_path + str(i) + '_viable.jpg'

    tumor_label = cv2.imread(tumor_label_path)
    viable_label = cv2.imread(viable_label_path)
    he, wi, _ = tumor_label.shape

    _, tumor_label = cv2.threshold(tumor_label, 250, 1, cv2.THRESH_BINARY)
    _, viable_label = cv2.threshold(viable_label, 250, 1, cv2.THRESH_BINARY)

    tumor_prediction_path = prediction_path + 'valid_{0}_prediction'.format(i) + '/tumor_result.jpg'
    viable_prediction_path = prediction_path + 'valid_{0}_prediction'.format(i) + '/viable_result.jpg'

    tumor_prediction = cv2.imread(tumor_prediction_path)
    viable_prediction = cv2.imread(viable_prediction_path)

    _, tumor_prediction = cv2.threshold(tumor_prediction, 250, 1, cv2.THRESH_BINARY)
    _, viable_prediction = cv2.threshold(viable_prediction, 250, 1, cv2.THRESH_BINARY)

    t = tumor_label[:, :, 0] * tumor_prediction[:he, :wi, 0]
    wrong_tumor_matrix = tumor_label[:, :, 0] - t
    t = viable_label[:, :, 0] * viable_prediction[:he, :wi, 0]
    wrong_viable_matrix = viable_label[:, :, 0] - t

    WSI_path = 'D:/PAIP_original_patch/Training_phase_1_{0:03d}'.format(i)

    a = glob.glob(WSI_path + '/*.svs')

    label_h, label_w = wrong_tumor_matrix.shape

    WSI_img = openslide.OpenSlide(a[0])
    WSI_w, WSI_h = WSI_img.level_dimensions[0]

    file_list = glob.glob(WSI_path + '/*.tif')
    file_name = file_list[0].split("\\")
    file_name = file_name[-1].split('.')
    file = file_name[0].split('_')

    if file[3] == 'viable':
        viable_path = file_list[0]
        tumor_path = file_list[1]
    else:
        tumor_path = file_list[0]
        viable_path = file_list[1]

    tumor_img = tifffile.imread(tumor_path)
    viable_img = tifffile.imread(viable_path)
    one=np.ones((1024,1024))
    for j in range(label_h):
        for k in range(label_w):
            if wrong_tumor_matrix[j, k] == 1 or wrong_viable_matrix[j, k] == 1:
                if 0 < 1024 * k or 1024 * k + 1024 < WSI_w or 0 < 1024 * j or 1024 * j + 1024 < WSI_h:
                    t_img = np.array(tumor_img[1024 * j:1024 * j + 1024, 1024 * k:1024 * k + 1024])
                    if t_img.shape == (1024, 1024):


                            patch = WSI_img.read_region((1024 * k, 1024 * j), 0, (1024, 1024)).convert('RGB')
                            s=np.asarray(patch)
                            if (s[:,:,0]*one).sum() <100:
                                continue
                            path = 'D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
                                i) + '/{0}_{1}.jpg'.format(1024 * j, 1024 * k)
                            patch.save(path)

                            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                            cv2.imwrite(
                                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_whole_label/' +
                                '/{0}_{1}.jpg'.format(1024 * j, 1024 * k), t_img)

                            t_img = np.array(viable_img[1024 * j:1024 * j + 1024, 1024 * k: 1024 * k + 1024])

                            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                            cv2.imwrite(
                                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_viable_label/' +
                                '/{0}_{1}.jpg'.format(1024 * j, 1024 * k), t_img)

                            # plt.subplot(1,3,1)
                            # plt.imshow(s)
                            # plt.subplot(1,3,2)
                            # plt.imshow(t_img)
                            # plt.subplot(1, 3, 3)
                            # plt.imshow(t_img)
                            # plt.show()
