import os
import cv2
import numpy as np
from tqdm import trange
import torch
import matplotlib.pyplot as plt
import tifffile
def mkdir_s(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_window(wsi_info, starting_point, window_size):
    wsi_w, wsi_h = wsi_info.level_dimensions[0]

    for y in trange(starting_point, wsi_h-1024,window_size):
        for x in range(starting_point, wsi_w-1024,window_size):
            window = wsi_info.read_region((x, y), 0, (window_size,window_size)).convert('RGB')
            window = np.array(window)
            yield x, y, window


def get_window_right(wsi_info, window_size,step_size):
    wsi_w, wsi_h = wsi_info.level_dimensions[0]
    x=wsi_w-window_size
    for y in range(0, wsi_h - window_size, step_size):
        window = wsi_info.read_region((x, y), 0, (window_size,window_size)).convert('RGB')
        window = np.array(window)
        yield x, y, window


def get_window_bottom(wsi_info, window_size,step_size):
    wsi_w, wsi_h = wsi_info.level_dimensions[0]
    y=wsi_h-window_size
    for x in range(0, wsi_w-window_size, step_size):
        window = wsi_info.read_region((x, y), 0, (window_size,window_size)).convert('RGB')
        window = np.array(window)
        yield x, y, window

def get_threshold_result(img):
    h, w, _ = img.shape

    R, G, B = cv2.split(img)
    _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
    _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
    _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)

    background_pixels = R * G * B
    forground_pixels = np.ones((h, w),dtype=np.uint8) - background_pixels

    return background_pixels, forground_pixels


def seg_prediction(result, s_net, img, start_w, start_h, foreground,window_size):
    w_h,w_w=(window_size,window_size)
    viable_batch = []
    for y in [0, w_h]:
        for x in [0, w_w]:
            viable_batch.append(img[y:y + w_h, x:x + w_w])
    viable_batch = np.array(viable_batch)
    viable_batch = torch.from_numpy(viable_batch)
    viable_batch = viable_batch.permute(0, 3, 1, 2)  # to NCHW
    viable_batch = viable_batch.to('cuda').float()

    logit = s_net(viable_batch)
    prob = torch.softmax(logit, dim=1)
    prob=prob.cpu()

    idx = 0

    for y in [0, w_h]:
        for x in [0, w_w]:
            result[start_h + y:start_h + y + w_h, start_w + x:start_w + x + w_w, 0] += \
                prob[idx][0].cpu().numpy() * foreground[y: y + w_w, x: + x + w_w]
            result[start_h + y:start_h + y + w_h, start_w + x:start_w + x + w_w, 1] += \
                prob[idx][1].cpu().numpy() * foreground[y: y + w_h, x: + x + w_w]
            idx += 1


    return result

def show_result(task1,task2):
    task1_result=tifffile.imread(task1[0])
    task2_result=tifffile.imread(task2[0])

    plt.subplot(231)

    plt.subplot(232)
    plt.imshow(task1_result)
    plt.subplot(233)
    plt.imshow(task2_result)

    task1_result = tifffile.imread(task1[0])
    task2_result = tifffile.imread(task2[0])

    plt.subplot(235)
    plt.subplot(235)
    plt.imshow(task1_result)
    plt.subplot(236)
    plt.imshow(task2_result)

    plt.show()
