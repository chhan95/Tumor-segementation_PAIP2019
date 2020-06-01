
import csv
import os
import random
import shutil

import numpy as np
import torch
from termcolor import colored


####
def check_manual_seed(seed):
    """ 
    If manual seed is not specified, choose a random one and communicate it to the user.
    """

    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('Using manual seed: {seed}'.format(seed=seed))
    return

####
def check_log_dir(log_dir):
    # check if log dir exist
    if os.path.isdir(log_dir):
        color_word = colored('WARNING', color='red', attrs=['bold', 'blink'])
        print('%s: %s exist!' % (color_word, colored(log_dir, attrs=['underline'])))
        while (True):
            print('Select Action: d (delete) / q (quit)', end='')
            key = input()
            if key == 'd':
                shutil.rmtree(log_dir)
                break
            elif key == 'q':
                exit()
            else:
                color_word = colored('ERR', color='red')
                print('---[%s] Unrecognize Characters!' % color_word)
    return

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

####
def bounding_box(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out 
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]
####