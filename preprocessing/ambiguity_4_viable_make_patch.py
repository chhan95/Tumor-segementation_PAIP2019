import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.external import tifffile
import os

whole_ambiguity_path = 'D:/PAIP_viable_tumor_patch/Ambiguity/'
original_WSI_path = 'D:/PAIP_original_patch/'
tumor_abs_path = 'D:/4_PAIP_cancer_label_patch/'

for i in range(1, 21):
    WSI_path = original_WSI_path + 'Training_phase_1_{0:03d}'.format(i)
    a = glob.glob(WSI_path + '/*.svs')
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

    amboguity_path = whole_ambiguity_path + 'Training_phase_1_{0:03d}'.format(i)
    b = glob.glob(amboguity_path + '/*.jpg')

    for j in b:
        filename = b[-1].split('\\')
        filename = filename[-1].split('.')
        h, w = filename[0].split('_')


        patch = WSI_img.read_region((1024 * int(h), 1024 * int(w)), 0, (1024, 1024)).convert('RGB')
        path = 'D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
            i) + '/{0}_{1}.jpg'.format(1024 * int(h), 1024 * int(w))

        patch.save(path)

        for y in [-1, 1]:
            for x in [-1, 1]:
                start_y = 1024 * int(h) + y * 512
                start_x = 1024 * int(w) + x * 512

                t_img = np.array(tumor_img[start_y:start_y + 1024, start_x:start_x + 1024])
                if start_x < 0 or start_x > WSI_w or start_y < 0 or start_y > WSI_h:
                    continue

                patch = WSI_img.read_region((start_x, start_y), 0, (1024, 1024)).convert('RGB')
                path = 'D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
                    i) + '/{0}_{1}.jpg'.format(start_y, start_x)
                patch.save(path)

                _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                cv2.imwrite(
                    tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_whole_label/' +
                    '/{0}_{1}.jpg'.format(start_y, start_x), t_img)

                t_img = np.array(viable_img[start_y:start_y + 1024, start_x:start_x + 1024])

                _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                cv2.imwrite(
                    tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_viable_label/' +
                    '/{0}_{1}.jpg'.format(start_y, start_x), t_img)

for i in range(21, 51):
    WSI_path = original_WSI_path + 'Training_phase_2_{0:03d}'.format(i)
    a = glob.glob(WSI_path + '/*.svs')
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

    amboguity_path = whole_ambiguity_path + 'Training_phase_2_{0:03d}'.format(i)
    b = glob.glob(amboguity_path + '/*.jpg')

    for j in b:
        filename = b[-1].split('\\')
        filename = filename[-1].split('.')
        h, w = filename[0].split('_')

        patch = WSI_img.read_region((1024 * int(h), 1024 * int(w)), 0, (1024, 1024)).convert('RGB')
        path = 'D:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(
            i) + '/{0}_{1}.jpg'.format(1024 * int(h), 1024 * int(w))

        patch.save(path)

        for y in [-1, 1]:
            for x in [-1, 1]:
                start_y = 1024 * int(h) + y * 512
                start_x = 1024 * int(w) + x * 512

                t_img = np.array(tumor_img[start_y:start_y + 1024, start_x:start_x + 1024])
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
