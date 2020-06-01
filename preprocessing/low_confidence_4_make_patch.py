import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.external import tifffile
import os

label_path = 'D:/WSI_label/'
prediction_path = 'D:/WSI_prediction/'
tumor_abs_path = 'D:/4_PAIP_cancer_label_patch/'
for i in range(21, 51):

    if not os.path.exists('D:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('D:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('D:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('D:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('D:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(i)):
        os.mkdir('D:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(i))

    tumor_co_path = prediction_path + 'valid_{0}_prediction/tumor_confidence.jpg'.format(i)
    viable_co_path = prediction_path + 'valid_{0}_prediction/viable_confidence.jpg'.format(i)

    tumor_co = cv2.imread(tumor_co_path)
    viable_co = cv2.imread(viable_co_path)

    _, tumor_co = cv2.threshold(tumor_co, 250, 1, cv2.THRESH_BINARY)
    _, viable_co = cv2.threshold(viable_co, 250, 1, cv2.THRESH_BINARY)

    tumor_co = tumor_co[:, :, 0]
    viable_co = viable_co[:, :, 0]

    WSI_path = 'D:/PAIP_original_patch/Training_phase_2_{0:03d}'.format(i)

    a = glob.glob(WSI_path + '/*.svs')

    label_h, label_w = viable_co.shape

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

    for j in range(label_h):
        for k in range(label_w):
            if tumor_co[j, k] == 1 or viable_co[j, k] == 1:

                t_img = np.array(tumor_img[1024 * j:1024 * j + 1024, 1024 * k:1024 * k + 1024])
                if t_img.shape == (1024, 1024):
                    patch = WSI_img.read_region((1024 * j, 1024 * k), 0, (1024, 1024)).convert('RGB')
                    path = 'D:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(
                        i) + '/{0}_{1}.jpg'.format(1024 * j, 1024 * k)

                    patch.save(path)

                    _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(
                        tumor_abs_path + 'Training_phase_2_{0:03d}'.format(i) + '_whole_label/' +
                        '/{0}_{1}.jpg'.format(1024 * j, 1024 * k), t_img)

                    t_img = np.array(viable_img[1024 * j:1024 * j + 1024, 1024 * k:1024 * k + 1024])
                    _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(
                        tumor_abs_path + 'Training_phase_2_{0:03d}'.format(i) + '_viable_label/' +
                        '/{0}_{1}.jpg'.format(1024 * j, 1024 * k), t_img)

                for y in [-1, 1]:
                    for x in [-1, 1]:
                        start_y = 1024 * j + y * 512
                        start_x = 1024 * k + x * 512

                        t_img = np.array(tumor_img[start_y:start_y + 1024, start_x:start_x + 1024])
                        if t_img.shape != (1024, 1024):
                            continue

                        if start_x < 0 or start_x > WSI_w or start_y < 0 or start_y > WSI_h:
                            continue

                        patch = WSI_img.read_region((start_x, start_y), 0, (1024, 1024)).convert('RGB')
                        path = 'D:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(
                            i) + '/{0}_{1}.jpg'.format(start_y, start_x)

                        patch.save(path)

                        _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                        cv2.imwrite(
                            tumor_abs_path + 'Training_phase_2_{0:03d}'.format(i) + '_whole_label/' +
                            '/{0}_{1}.jpg'.format(start_y, start_x), t_img)

                        t_img = np.array(viable_img[start_y:start_y + 1024, start_x:start_x + 1024])
                        _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                        cv2.imwrite(
                            tumor_abs_path + 'Training_phase_2_{0:03d}'.format(i) + '_viable_label/' +
                            '/{0}_{1}.jpg'.format(start_y, start_x), t_img)
