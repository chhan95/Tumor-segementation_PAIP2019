
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import ndimage
from skimage.morphology import (binary_dilation, remove_small_holes,
                                remove_small_objects)

def center_pad_to(img, h, w):
    shape = img.shape

    diff_h = h - shape[0]
    padt = diff_h // 2
    padb = diff_h - padt

    diff_w = w - shape[1]
    padl = diff_w // 2
    padr = diff_w - padl

    if len(img.shape) == 3:
        img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), 
                        'constant', constant_values=255)
    else:
        img = np.lib.pad(img, ((padt, padb), (padl, padr)), 
                        'constant', constant_values=255)

    return img

tma_code = '11S-1_1(x400)'
model_code = 'pytorch/v1.0.0.4'
pred_core_dir = '/mnt/dang/infer/KBSMC/BREAST/%s/%s/' % (model_code, tma_code)
orig_core_dir = '/mnt/dang/data/KBSMC/PROSTATE/%s/imgs/' % tma_code

# tma_code = 'BT140001'
# model_code = 'pytorch/v1.0.0.4'
# pred_core_dir = '/mnt/dang/infer/KBSMC/BREAST/%s/%s/' % (model_code, tma_code)
# orig_core_dir = '/mnt/dang/data/KBSMC/BREAST/%s/imgs/' % tma_code

canvas_path = '/mnt/dang/infer/KBSMC/BREAST/'
file_list = glob.glob(pred_core_dir + '/*.png')
file_list.sort() # ensure same order [1]

# prostate
core_assume_shape = [2048, 2048] # core bigger will break code
canvas = np.full((core_assume_shape[0] * 12, 
                  core_assume_shape[1] * 16, 3), 255, np.uint8)

# a white canvas
# core_assume_shape = [2752, 2752] # core bigger will break code
# canvas = np.full((core_assume_shape[0] * 6, 
#                   core_assume_shape[1] * 10, 3), 255, np.uint8)

stat_matrix = np.zeros([12, 16], dtype=float)

row_idx = 0
col_idx = 0
pattern = "(([A-Z])(\d+))"
for filename in file_list[:]:
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]

    name_code = re.match(pattern, basename)
    row_idx = ord(name_code.group(2)) - ord('A') 
    col_idx = int(name_code.group(3)) - 1
    # print(basename, row_idx, col_idx)

    row_start = row_idx * core_assume_shape[1]
    col_start = col_idx * core_assume_shape[0]
    row_end = (row_idx + 1) * core_assume_shape[1]
    col_end = (col_idx + 1) * core_assume_shape[0]

    # center paste the core onto the canvas
    # orig_core = cv2.imread(orig_core_dir + basename + '.jpg')
    # orig_core = cv2.resize(orig_core, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    # orig_core = center_pad_to(orig_core, 2752, 2752)

    # prostate
    orig_core = cv2.imread(orig_core_dir + basename + '.jpg')
    orig_core = cv2.resize(orig_core, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    orig_core = center_pad_to(orig_core, 2048, 2048)

    orig_gray = cv2.cvtColor(orig_core, cv2.COLOR_RGB2GRAY)
    thval, thmap = cv2.threshold(orig_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  

    # HACK: for A02 BT140003
    # if 'A02' not in basename and 'BT140003' not in basename:     
    #     thmap = np.array(thmap == 0)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = remove_small_objects(thmap == 0, min_size=400*400)

    pred_core = np.load(pred_core_dir + basename + '.npy')
    pred_core = center_pad_to(pred_core, 2048, 2048)

    pred_core[thmap > 0] = 0 # set the empty background

    pred_mask = np.copy(pred_core)
    pred_mask = np.array(pred_mask > 0.5, np.bool)
    pred_mask = remove_small_holes(pred_mask, 64)
    pred_mask = remove_small_objects(pred_mask, 64)
    pred_mask = binary_dilation(pred_mask)
    pred_mask = pred_mask.astype('int32')
    k = np.array([[1,  1,  1],
                  [1, -8,  1],
                  [1,  1,  1]])
    mask_contour = ndimage.convolve(pred_mask, k)
    mask_contour = np.array(mask_contour > 0)
    mask_contour = remove_small_objects(mask_contour, 32)
    mask_contour = binary_dilation(mask_contour)
    mask_contour = binary_dilation(mask_contour)

    orig_core = orig_core.astype('int32')
    orig_core[pred_mask > 0] -= 40
    orig_core[orig_core < 0]  = 0
    orig_core = orig_core.astype('uint8')

    ##
    orig_core_ch = orig_core[...,0]
    orig_core_ch[mask_contour > 0] = 0

    orig_core_ch = orig_core[...,1]
    orig_core_ch[mask_contour > 0] = 50

    orig_core_ch = orig_core[...,2]
    orig_core_ch[mask_contour > 0] = 255

    canvas[row_start : row_end, col_start : col_end] = orig_core

    ## stat
    pred_mask[thmap > 0] = 2 # this is the white background, hence ignore it
    epi_pix = float((pred_mask == 1).sum()) 
    str_pix = float((pred_mask == 0).sum())
    stat_matrix[row_idx, col_idx] = epi_pix / (str_pix + 1.0e-6)

print(stat_matrix)
cv2.imwrite('%s/%s/%s.jpg' % (canvas_path, model_code, tma_code), canvas)
