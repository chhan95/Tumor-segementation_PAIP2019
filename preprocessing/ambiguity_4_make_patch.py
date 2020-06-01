import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.external import tifffile
import os

whole_ambiguity_path = 'E:/PAIP_benign_cancer_patch/Ambiguity/'
original_WSI_path = 'E:/PAIP_original_patch/'
tumor_abs_path = 'E:/4_PAIP_cancer_label_patch/'



for i in range(1, 21):

    if not os.path.exists('E:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('E:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('D:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('E:/4_PAIP_cancer_label_patch/Training_phase_1_{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('D:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(i)):
        os.mkdir('E:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(i))

    WSI_path = original_WSI_path + 'Training_phase_1_{0:03d}'.format(i)
    a = glob.glob(WSI_path + '/*.svs')
    WSI_img = openslide.OpenSlide(a[0])
    WSI_w, WSI_h = WSI_img.level_dimensions[0]

    file_list=glob.glob(WSI_path+'/*.tif')
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
        filename=b[-1].split('\\')
        filename=filename[-1].split('.')
        h,w=filename[0].split('_')
        h=int(h)
        w=int(w)
        t_img = np.array(tumor_img[1024*h:1024*h + 1024, 1024 * w:1024 * w + 1024])
        if t_img.shape == (1024, 1024):
            patch = WSI_img.read_region((1024 * w, 1024*h), 0, (1024, 1024)).convert('RGB')
            path = 'E:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
                i) + '/{0}_{1}.jpg'.format(1024*h, 1024 * w)
            patch.save(path)

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_whole_label/' +
                '/{0}_{1}.jpg'.format(1024*h, 1024 * w), t_img)

            t_img = np.array(viable_img[1024*h:1024*h + 1024, 1024 * w:1024 * w + 1024])

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_viable_label/' +
                '/{0}_{1}.jpg'.format(1024*h,1024 * w), t_img)


        for y in [-1, 1]:
            for x in [-1, 1]:
                start_y = 1024 * int(h) + y * 512
                start_x = 1024 * int(w) + x * 512

                t_img = np.array(tumor_img[start_y:start_y + 1024, start_x:start_x + 1024])
                if start_x < 0 or start_x+1024 > WSI_w or start_y < 0 or start_y+1024 > WSI_h:
                    continue

                patch = WSI_img.read_region((start_x, start_y), 0, (1024, 1024)).convert('RGB')
                path = 'E:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
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

    if not os.path.exists('E:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('E:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('E:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('E:/4_PAIP_cancer_label_patch/Training_phase_2_{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('E:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(i)):
        os.mkdir('E:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(i))

    WSI_path = original_WSI_path + 'Training_phase_2_{0:03d}'.format(i)
    a = glob.glob(WSI_path + '/*.svs')
    WSI_img = openslide.OpenSlide(a[0])
    WSI_w, WSI_h = WSI_img.level_dimensions[0]

    file_list=glob.glob(WSI_path+'/*.tif')
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
        filename=b[-1].split('\\')
        filename=filename[-1].split('.')
        h,w=filename[0].split('_')
        h=int(h)
        w=int(w)
        t_img = np.array(tumor_img[1024 * h:1024 * h + 1024, 1024 * w:1024 * w + 1024])
        if t_img.shape == (1024, 1024):
            patch = WSI_img.read_region((1024 * w, 1024 * h), 0, (1024, 1024)).convert('RGB')
            path = 'E:/4_PAIP_original_patch/Training_phase_1_{0:03d}'.format(
                i) + '/{0}_{1}.jpg'.format(1024 * h,1024 * w)
            patch.save(path)

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_whole_label/' +
                '/{0}_{1}.jpg'.format(1024 * h, 1024 * w), t_img)

            t_img = np.array(viable_img[1024 * h:1024 * h + 1024,1024 * w:1024 * w + 1024])

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + 'Training_phase_1_{0:03d}'.format(i) + '_viable_label/' +
                '/{0}_{1}.jpg'.format(1024 * h, 1024 * w), t_img)

        for y in [-1, 1]:
            for x in [-1, 1]:
                start_y = 1024 * int(h) + y * 512
                start_x = 1024 * int(w) + x * 512

                t_img = np.array(tumor_img[start_y:start_y + 1024, start_x:start_x + 1024])
                if t_img.shape != (1024, 1024):
                    continue

                patch = WSI_img.read_region((start_x, start_y), 0, (1024, 1024)).convert('RGB')
                path = 'E:/4_PAIP_original_patch/Training_phase_2_{0:03d}'.format(
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
