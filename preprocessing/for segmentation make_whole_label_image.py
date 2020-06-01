from skimage.external import tifffile
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from PIL import Image
import openslide


tumor_abs_path = 'dataset/cancer_viable_label_patch/'


for i in range(1, 51):
    if not os.path.exists('dataset/cancer_viable_label_patch_for_segmentation_2048/{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('dataset/cancer_viable_label_patch_for_segmentation_2048/{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('dataset/cancer_viable_label_patch_for_segmentation_2048/{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('dataset/cancer_viable_label_patch_for_segmentation_2048/{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('dataset/original_patch_for_segmentation_2048/{0:03d}'.format(i) ):
        os.mkdir('dataset/original_patch_for_segmentation_2048/{0:03d}'.format(i))

    wsi_list=glob.glob('dataset/svs_tif/{0:03d}'.format(i)+'/*.svs')
    file_list = glob.glob('dataset/svs_tif/{0:03d}'.format(i) + '/*.tif')
    file_name = file_list[0].split("\\")
    file_name = file_name[-1].split('.')
    file = file_name[0].split('_')



    wsi_img=openslide.OpenSlide(wsi_list[0])
    if file[3] == 'viable':
        viable_path = file_list[0]
        tumor_path = file_list[1]
    else:
        tumor_path = file_list[0]
        viable_path = file_list[1]

    tumor_img = tifffile.imread(tumor_path)
    viable_img = tifffile.imread(viable_path)
    h, w = tumor_img.shape

    for j in range(0, h, 2048):
        for k in range(0, w, 2048):

            t_img = np.array(tumor_img[j:j + 2048, k:k + 2048])

            if t_img.shape != (2048, 2048):
                continue
            patch=wsi_img.read_region((k,j),0,(2048,2048)).convert('RGB')
            path='dataset/original_patch/{0:03d}/{1}_{2}.png'.format(i,j,k)
            patch.save(path)


            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + '{0:03d}'.format(i) + '_whole_label/' + str(j) + '_' + str(
                    k ) + '.png', t_img)

            t_img = np.array(viable_img[j:j + 2048, k:k + 2048])

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + '{0:03d}'.format(i) + '_viable_label/' + str(j) + '_' + str(
                    k ) + '.png', t_img)


