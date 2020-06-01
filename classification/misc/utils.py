
import os
import shutil
import random
import numpy as np

####
def color_mask(a, r, g, b):
    ch_r = a[...,0] == r
    ch_g = a[...,1] == g
    ch_b = a[...,2] == b
    return ch_r & ch_g & ch_b
####
def normalize(mask, dtype=np.uint8):
    return (255 * mask / np.amax(mask)).astype(dtype)
####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax
####
def cropping_center(x, crop_shape, batch=False):   
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:,h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]        
    return x
#####
# to make it easier for visualization
def randomize_label(label_map):
    label_list = np.unique(label_map)
    label_list = label_list[1:] # exclude the background
    label_rand = list(label_list) # dup first cause shuffle is done in place
    random.shuffle(label_rand)
    new_map = np.zeros(label_map.shape, dtype=label_map.dtype)
    for idx, lab_id in enumerate(label_list):
        new_map[label_map == lab_id] = label_rand[idx] + 50      
    return new_map
#####
def rm_n_mkdir(dir):
    if (os.path.isdir(dir)):
        shutil.rmtree(dir)
    os.makedirs(dir)