from skimage.external import tifffile
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from PIL import Image
import openslide


tumor_abs_path = '/media/disk/han/dataset/cancer_viable_label_patch/'


for i in range(1,51):
    if not os.path.exists('/media/disk/han/dataset/cancer_viable_label_patch/{0:03d}'.format(i) + '_whole_label'):
        os.mkdir('/media/disk/han/dataset/Segmentation_whole/{0:03d}'.format(i) + '_whole_label')
    if not os.path.exists('/media/disk/han/dataset/Segmentation_whole/{0:03d}'.format(i) + '_viable_label'):
        os.mkdir('/media/disk/han/dataset/Segmentation_whole/{0:03d}'.format(i) + '_viable_label')
    if not os.path.exists('/media/disk/han/dataset/original_patch/{0:03d}'.format(i) ):
        os.mkdir('/media/disk/han/dataset/original_patch/{0:03d}'.format(i))

    wsi_list=glob.glob('/media/disk/han/dataset/svs_tif/{0:03d}'.format(i)+'/*.svs')
    wsi_list2=glob.glob('/media/disk/han/dataset/svs_tif/{0:03d}'.format(i)+'/*.SVS')
    if len(wsi_list2)>0:
        wsi_list.append(wsi_list2[0])
    file_list = glob.glob('/media/disk/han/dataset/svs_tif/{0:03d}'.format(i) + '/*.tif')
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

    for j in range(0, h, 1024):
        for k in range(0, w, 1024):

            t_img = np.array(tumor_img[j:j + 1024, k:k + 1024])

            if t_img.shape != (1024, 1024):
                continue
            patch=wsi_img.read_region((k,j),0,(1024,1024)).convert('RGB')
            path='/media/disk/han/dataset/original_patch/{0:03d}/{1}_{2}.png'.format(i,j,k)
            patch.save(path)


            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + '{0:03d}'.format(i) + '_whole_label/' + str(j) + '_' + str(
                    k ) + '.png', t_img)

            t_img = np.array(viable_img[j:j + 1024, k:k + 1024])

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                tumor_abs_path + '{0:03d}'.format(i) + '_viable_label/' + str(j) + '_' + str(
                    k ) + '.png', t_img)


