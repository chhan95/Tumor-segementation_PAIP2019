import cv2
import openslide
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.external import tifffile
import os

root_path = 'D:/'
original_image_path = root_path + 'PAIP_original_patch/'

tumor_abs_path = 'D:/new_4_label/'
for i in range(21, 51):

    if not os.path.exists('D:/new_4_label/Training_phase_2_{0:03d}_whole_label'.format(i)):
        os.mkdir('D:/new_4_label/Training_phase_2_{0:03d}_whole_label'.format(i))
    if not os.path.exists('D:/new_4/Training_phase_2_{0:03d}'.format(i)):
        os.mkdir('D:/new_4/Training_phase_2_{0:03d}'.format(i))

    WSI_path = 'D:/PAIP_original_patch/Training_phase_2_{0:03d}'.format(i)

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

    file_list = glob.glob(original_4_image_path + 'Training_phase_2_{0:03d}/*.jpg'.format(i))

    for file in file_list:
        file_sp = file.split('\\')
        file_name = file_sp[-1].split('.')
        h, w = file_name[0].split('_')
        h = int(h)
        w = int(w)

        print('{0} {1}'.format(h, w))

        patch = WSI_img.read_region((w, h), 0, (1024, 1024)).convert('RGB')
        path = 'D:/new_4/Training_phase_2_{0:03d}'.format(
            i) + '/{0}_{1}.jpg'.format(h, w)
        patch.save(path)

        t_img = np.array(tumor_img[h:h + 1024, w: w + 1024])
        _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(
            tumor_abs_path + 'Training_phase_2_{0:03d}_whole_label'.format(i)  +
            '/{0}_{1}.jpg'.format(h, w), t_img)

        for j in [-512, 512]:
            for k in [-512, 512]:
                t_img = np.array(tumor_img[h + j:h + j + 1024, w + k: w + k + 1024])
                if t_img.shape != (1024, 1024):
                    continue

                patch = WSI_img.read_region((w + k, h + j), 0, (1024, 1024)).convert('RGB')
                path = 'D:/new_4/Training_phase_2_{0:03d}'.format(
                    i) + '/{0}_{1}.jpg'.format(h + j, w + k)
                patch.save(path)

                _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
                cv2.imwrite(
                    tumor_abs_path + 'Training_phase_2_{0:03d}_whole_label'.format(i) +
                    '/{0}_{1}.jpg'.format(h + j, w + k), t_img)
