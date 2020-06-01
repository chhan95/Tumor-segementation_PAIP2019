
import re

import cv2
import json
import numpy as np

import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import (binary_dilation, binary_erosion,
                                remove_small_holes,
                                remove_small_objects)
import openslide

import misc

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
####
scale_factor = 0.5
wsi_name = 'BT140004'
wsi_core_path = '/media/vqdang/Data/Workspace/KBSMC/BREAST/%s_core.txt' % wsi_name
pred_core_dir = '/media/vqdang/Data_2/dang/infer/kbsmc/breast/%s' % wsi_name 
orig_core_dir = '/media/vqdang/Data/Workspace/KBSMC/BREAST/%s/imgs/' % wsi_name

core_map = core_map[wsi_name]

wsi = cv2.imread('output/orig/%s.jpg' % wsi_name)
wsi = cv2.cvtColor(wsi, cv2.COLOR_BGR2RGB)

with open(wsi_core_path) as f:
    tma_info = json.load(f)

pattern = "(([A-Z])(\d+))"
stat_matrix = np.zeros([6, 10], dtype=np.float32)
for core_name in tma_info:        

    ########################################################################
    vertex_list = []
    core_roi = tma_info[core_name]
    for point in core_roi:
        pos_x = float(point['X'])
        pos_y = float(point['Y'])
        vertex_list.append([pos_x, pos_y])
    vertex_list = np.array(vertex_list) * scale_factor + 0.5
    vertex_list = vertex_list.astype('int32')

    min_x = int((np.amin(vertex_list[...,0]) * scale_factor) + 0.5)
    max_x = int((np.amax(vertex_list[...,0]) * scale_factor) + 0.5)
    min_y = int((np.amin(vertex_list[...,1]) * scale_factor) + 0.5)
    max_y = int((np.amax(vertex_list[...,1]) * scale_factor) + 0.5)

    # print(min_x, max_x, min_y, max_y)

    # shifting the coordinates to 0,0 at top corner
    vertex_list[...,0] -= min_x
    vertex_list[...,1] -= min_y

    # extract the core region in wsi
    wsi_core = wsi[min_y:max_y, min_x:max_x]

    ####
    core_gray = cv2.cvtColor(wsi_core, cv2.COLOR_RGB2GRAY)
    thval, thmap = cv2.threshold(core_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  

    # for v1
    # if 'A02' != core_name and 'BT140003' != wsi_name:     
    #     thmap = np.array(thmap == 0)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = binary_dilation(thmap)
    #     thmap = remove_small_objects(thmap == 0, min_size=200*200)
    #     thmap = np.array(thmap > 0, dtype=np.uint8)
        # thmap = np.array(thmap == 0, dtype=np.uint8)
        # plt.subplot(1,2,1)
        # plt.imshow(wsi_core)
        # plt.subplot(1,2,2)
        # plt.imshow(thmap)
        # plt.show()
        # continue

    # for v3
    # threshold within stroma is too strong so erode it abit so that the
    # later ratio is not too exaggerate
    # thmap = binary_erosion(thmap)

    core_pred = np.load('%s/%s.npy' % (pred_core_dir, core_name))
    # NOTE: align shape with actual shape extracted from wsi
    core_pred = misc.cropping_center(core_pred, wsi_core.shape)
    core_pred[thmap > 0] = 0 # set the empty background

    pred_mask = np.copy(core_pred)
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

    orig_core = np.copy(wsi_core)
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

    # now overlaid
    wsi[min_y:max_y, min_x:max_x] = orig_core

    ##
    # print(core_name)
    if core_name in core_map:
        core_name = core_map[core_name]
    name_code = re.match(pattern, core_name)
    # map to correct layout
    row_idx = ord(name_code.group(2)) - ord('A') 
    col_idx = int(name_code.group(3)) - 1
    ##
    pred_mask[thmap > 0] = 2 # this is the white background, hence ignore it
    ##
    epi_pix = float((pred_mask == 1).sum()) 
    str_pix = float((pred_mask == 0).sum())
    epi_str_ratio = epi_pix / (str_pix + 1.0e-6)
    # epi_str_ratio = epi_pix / (str_pix + epi_pix + 1.0e-6)
    stat_matrix[row_idx, col_idx] = epi_str_ratio

    # if core_name == 'C03':
    #     plt.subplot(1,2,1)
    #     plt.imshow(orig_core)
    #     plt.subplot(1,2,2)
    #     plt.imshow(pred_mask)
    #     plt.title('%.5f' % epi_str_ratio)
    #     plt.show()
    #     exit()
    ########################################################################
    # extra stuff, drawing the box for ID and stroma ratio directly onto the wsi
    # text = '%s: %0.5f' % (core_name, epi_str_ratio)
    # draw_area = wsi[min_y-300:min_y, min_x:min_x+1500]
    # cv2.putText(draw_area,  text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, cv2.LINE_AA)
    # wsi[min_y-300:min_y, min_x:min_x+1500] = draw_area
    ####
print(stat_matrix)
# wsi = cv2.cvtColor(wsi, cv2.COLOR_RGB2BGR)
# cv2.imwrite('output/v1/%s_overlaid.jpg' % wsi_name, wsi)