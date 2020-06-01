
import shutil
import os
import csv
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ####
# def color_mask(a, r, g, b):
#     ch_r = a[...,0] == r
#     ch_g = a[...,1] == g
#     ch_b = a[...,2] == b
#     return ch_r & ch_g & ch_b
# ####
# def rm_n_mkdir(dir):
#     if (os.path.isdir(dir)):
#         shutil.rmtree(dir)
#     os.makedirs(dir)
# ####

# tma_id = 'BT140004'
# imgs_dir = '/mnt/dang/data/KBSMC/BREAST/%s/imgs/' % tma_id
# anns_dir = '/mnt/dang/data/KBSMC/BREAST/%s/anns/' % tma_id
# ####
# out_dir = '/mnt/dang/train/KBSMC/BREAST/orig/%s/' % tma_id

# cores_dict = {
#     'BT140001': ['A03', 'A09', 'B08', 'D05', 'D07', 'D10', 'E02'],
#     'BT140002': ['A02', 'A04', 'A05', 'A07', 'A09', 'A10', 'B09', 'C04',
#                  'C07', 'D02', 'D03', 'D04', 'D06', 'D07', 'E04'],
#     'BT140004': ['B05', 'C01', 'C07', 'C09', 'D09', 'D10', 'E08']}

# file_list = glob.glob(anns_dir + '/*.png')
# file_list.sort() # ensure same order [1]

# # rm_n_mkdir(out_dir)

# for filename in file_list: # png for base
#     filename = os.path.basename(filename)
#     basename = filename.split('.')[0]

#     if basename not in cores_dict[tma_id]:
#         continue
#     print(basename, end=' ', flush=True)

#     img = cv2.imread(imgs_dir + basename + '.jpg')
#     ann = cv2.imread(anns_dir + basename + '.png')

#     # sync shape
#     min_h = min(img.shape[0], ann.shape[0])
#     min_w = min(img.shape[1], ann.shape[1])
#     img = img[:min_h, :min_w]
#     ann = ann[:min_h, :min_w]

#     img = cv2.resize(img, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
#     ann = cv2.resize(ann, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
#     ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

#     # proc the annotation colors
#     epi_msk = color_mask(ann,  51,  26, 128)
#     str_msk = color_mask(ann, 204, 255, 204)
#     out_msk = color_mask(ann,   0,   0,   0)
#     ann = np.zeros(ann.shape[:2], np.uint8)
#     ann[epi_msk] = 2
#     ann[str_msk] = 1
#     ann[out_msk] = 1

#     # plt.imshow(ann)
#     # plt.show()
#     # continue

#     cv2.imwrite("%s/%s.jpg" % (out_dir, basename), img)
#     np.save("%s/%s.npy" % (out_dir, basename), ann)
#     print('---FINISH')

#---- For renaming the core filenames
####
core_map = {
    'BT140001' : {
        'A01' : 'A01', 'B01' : 'B06', 'C01' : 'C06', 'D01' : 'D08', 'E01' : 'F01',
        'A02' : 'A02', 'B02' : 'B07', 'C02' : 'C07', 'D02' : 'D09', 'E02' : 'F02',
        'A03' : 'A03', 'B03' : 'B08', 'C03' : 'C08', 'D03' : 'E01', 'E03' : 'F04',
        'A04' : 'A05', 'B04' : 'B09', 'C04' : 'C09', 'D04' : 'E02', 'E04' : 'F05',
        'A05' : 'A06', 'B05' : 'B10', 'C05' : 'D01', 'D05' : 'E03', 'E05' : 'F06', 
        'A06' : 'A08', 'B06' : 'C01', 'C06' : 'D02', 'D06' : 'E04', 'E06' : 'F07',
        'A07' : 'A09', 'B07' : 'C02', 'C07' : 'D03', 'D07' : 'E06', 'E07' : 'F08',
        'A08' : 'A10', 'B08' : 'C03', 'C08' : 'D05', 'D08' : 'E07', 'E08' : 'F10',
        'A09' : 'B01', 'B09' : 'C04', 'C09' : 'D06', 'D09' : 'E09', 
        'A10' : 'B02', 'B10' : 'C05', 'C10' : 'D07', 'D10' : 'E10',  
    },
    'BT140002' : {
        'A01' : 'A03', 'B01' : 'B07', 'C01' : 'C05', 'D01' : 'D09', 'E01' : 'E09',
        'A02' : 'A04', 'B02' : 'B08', 'C02' : 'C06', 'D02' : 'C10', 'E02' : 'E10',
        'A03' : 'A05', 'B03' : 'B09', 'C03' : 'C07', 'D03' : 'D10', 'E03' : 'F01',
        'A04' : 'A06', 'B04' : 'B10', 'C04' : 'C08', 'D04' : 'E01', 'E04' : 'F08',
        'A05' : 'A07', 'B05' : 'B01', 'C05' : 'D03', 'D05' : 'E02', 'E05' : 'F04', 
        'A06' : 'A09', 'B06' : 'B03', 'C06' : 'D04', 'D06' : 'E03', 'E06' : 'F06',
        'A07' : 'A10', 'B07' : 'C01', 'C07' : 'D05', 'D07' : 'E05', 'E07' : 'F07',
        'A08' : 'B02', 'B08' : 'C02', 'C08' : 'D06', 'D08' : 'E06', 'E08' : 'A02',
        'A09' : 'B04', 'B09' : 'C03', 'C09' : 'D07', 'D09' : 'E07', 
        'A10' : 'B06', 'B10' : 'C04', 'C10' : 'D08', 'D10' : 'E08',  
    },
    'BT140003' : {
        'F01' : 'F02',
        'F02' : 'F03',
        'F03' : 'F04',
        'F04' : 'F05',
        'F05' : 'F06',
        'F06' : 'F07',
        'F07' : 'F08',
        'F08' : 'F09',
        'F09' : 'F10',
        'F10' : 'F01',
    },
    'BT140004' : {},
    'BT140005' : {}
}

tma_code = 'BT140003'
# orig_dir and renamed_dir must be different else 
# the files will be overwritten and lost during the process
orig_dir = '/media/vqdang/Data_3/infer/KBSMC/proc/%s/' % tma_code
renamed_dir = '/media/vqdang/Data_3/infer/KBSMC/proc_x/%s/' % tma_code

import glob
import shutil

file_list = glob.glob('%s/*.mat' % orig_dir)
file_list.sort()

if not os.path.isdir(renamed_dir):
    os.makedirs(renamed_dir) 

map_name = core_map[tma_code]
for filename in file_list[:]:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]

    old_name = '%s/%s.mat' % (orig_dir, basename)
    if basename in map_name:
        new_name = '%s/%s.mat' % (renamed_dir, map_name[basename])
    else:
        new_name = '%s/%s.mat' % (renamed_dir, basename)
    print(old_name, new_name)
    shutil.move(old_name, new_name)
exit()

# #---------- For merging Nuclei and Epithelium vs Stroma prediction
# from scipy import io as sio
# from scipy import ndimage
# from skimage.morphology import (binary_dilation, remove_small_holes,
#                                 remove_small_objects)

# tma_code = 'BT140005'
# img_dir = '/media/vqdang/Data/Workspace/KBSMC/BREAST/%s/imgs_x/' % tma_code
# ign_dir = '/media/vqdang/Data/Workspace/KBSMC/BREAST/%s/anns_x/' % tma_code
# epi_dir = '/media/vqdang/Data_3/infer/KBSMC/BREAST/epi_str/%s/' % (tma_code)
# nuc_dir = '/media/vqdang/Data_3/infer/KBSMC/nuc_xy/%s/_proc/' % (tma_code)
# out_dir = '/media/vqdang/Data_3/infer/KBSMC/proc/%s/' % (tma_code)

# print(img_dir)
# file_list = glob.glob('%s/*.jpg' % img_dir)
# file_list.sort()

# if not os.path.isdir(out_dir):
#     os.makedirs(out_dir) 

# for filename in file_list[:]:
#     filename = os.path.basename(filename)
#     basename = filename.split('.')[0]
#     print(basename)

#     img = cv2.imread('%s/%s.jpg' % (img_dir, basename))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     ign = cv2.imread('%s/%s.png' % (ign_dir, basename))
#     ign = cv2.cvtColor(ign, cv2.COLOR_BGR2GRAY)

#     nuc = sio.loadmat('%s/%s.mat' % (nuc_dir, basename))['inst_map']
#     epi = np.load('%s/%s.npy' % (epi_dir, basename))[...,1]

#     # for removal of artifact areas
#     img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     thval, thmap = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  

#     # if 'A02' not in basename and 'BT140003' not in basename:     
#     #     thmap = np.array(thmap == 0)
#     #     thmap = binary_dilation(thmap)
#     #     thmap = binary_dilation(thmap)
#     #     thmap = binary_dilation(thmap)
#     #     thmap = binary_dilation(thmap)
#     #     thmap = remove_small_objects(thmap == 0, min_size=400*400*16)

#     # * align to x40 when processing
#     epi = cv2.resize(epi, (thmap.shape[1], thmap.shape[0]), interpolation=cv2.INTER_CUBIC)
#     epi[(thmap > 0) | (ign > 0)] = 0 # set the empty background
#     epi = np.copy(epi)
#     epi = np.array(epi > 0.5, np.bool)
#     epi = remove_small_holes(epi, 64*4)
#     epi = remove_small_objects(epi, 64*4)
#     epi = binary_dilation(epi)
#     epi = epi.astype('int8') # saving space

#     # setting the label
#     epi += 1 # 0: ignore, 1: stroma, 2: epithelium
#     epi[(thmap > 0) | (ign > 0)] = 0

#     # * naive threshold removal, hope it doesnt cross over any
#     # * detected instance
#     nuc[(thmap > 0) | (ign > 0)] = 0

#     sio.savemat('%s/%s.mat' % (out_dir, basename), 
#                 {'epi_str' : epi, 'nuc' : nuc})

#     # plt.subplot(1,3,1)
#     # plt.imshow(img)
#     # plt.subplot(1,3,2)
#     # plt.imshow(epi)
#     # plt.subplot(1,3,3)
#     # plt.imshow(nuc)
#     # plt.show()