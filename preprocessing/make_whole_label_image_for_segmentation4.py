from skimage.external import tifffile
import matplotlib.pyplot as plt
import cv2
import glob
import os
import numpy as np
from PIL import Image
import openslide

#dataset_path='/media/disk/han/dataset/seg_viable/'
tumor_abs_path = '/media/disk/han/dataset/Segmentation_whole/'


for i in range(1,51):
    if not os.path.exists('/media/disk/han/dataset/2048/viable_label/{0:03d}'.format(i) ):
        os.mkdir('/media/disk/han/dataset/2048/viable_label/{0:03d}'.format(i) )
    if not os.path.exists('/media/disk/han/dataset/2048/whole_label{0:03d}'.format(i) ):
        os.mkdir('/media/disk/han/dataset/2048/whole_label/{0:03d}'.format(i) )
    if not os.path.exists('/media/disk/han/dataset/2048/original_image/{0:03d}'.format(i) ):
        os.mkdir('/media/disk/han/dataset/2048/original_image/{0:03d}'.format(i))

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

    for j in range(0, h, 2048):
        for k in range(0, w, 2048):

            t_img = np.array(tumor_img[j:j + 2048, k:k + 2048])

            if t_img.shape != (2048, 2048):
                continue

            patch=wsi_img.read_region((k,j),0,(2048,2048)).convert('RGB')
            a=np.array(patch)
            B, G, R = cv2.split(a)
            _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
            _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
            _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)

            background_label_img = B * G * R
            forground_label_img = np.ones((2048, 2048)) - background_label_img

            if forground_label_img.sum() < 209715:
                continue

            path='/media/disk/han/dataset/2048/original_image/{0:03d}/{1}_{2}.png'.format(i,j,k)
            patch.save(path)

            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                 '/media/disk/han/dataset/2048/whole_label/{0:03d}/'.format(i) + str(j) + '_' + str(
                    k ) + '.png', t_img)

            t_img = np.array(viable_img[j:j + 2048, k:k + 2048])
            _, t_img = cv2.threshold(t_img, 0.5, 255, cv2.THRESH_BINARY)
            cv2.imwrite(
                 '/media/disk/han/dataset/2048/viable_label/{0:03d}/'.format(i) + str(j) + '_' + str(k ) + '.png', t_img)


