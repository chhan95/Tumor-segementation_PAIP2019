from skimage.filters import try_all_threshold
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import openslide
from scipy.ndimage import label, generate_binary_structure

from skimage.morphology import remove_small_objects
import tifffile

file_list = glob.glob('/media/disk/han/dataset/testset/testset/*.svs')
for filename in file_list:
    print(filename[-7:-4])

    original_img = openslide.OpenSlide(filename)

    t_img = tifffile.imread('/media/disk/han/dataset/testset/256/{0}_t.tif'.format(filename[-7:-4]))
    v_img = tifffile.imread('/media/disk/han/dataset/testset/256/{0}.tif'.format(filename[-7:-4]))

    h, w = t_img.shape

    original_result = np.zeros((h, w))

    t_img = np.array(t_img)
    v_img = np.array(v_img)

    for i in range(0, h - 1024, 1024):

        for j in range(0, w - 1024, 1024):
            patch = original_img.read_region((j, i), 0, (1024, 1024)).convert('RGB')
            patch=np.array(patch)
            R, G, B = cv2.split(patch)
            _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
            _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
            _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)

            background_label_img = R * B * G
            forground_label_img = np.ones((1024, 1024)) - background_label_img
            original_result[i:i + 1024, j:j + 1024] = forground_label_img

    for i in range(0, h - 1024, 1024):
        patch = original_img.read_region((w - 1024, i), 0, (1024, 1024)).convert('RGB')
        patch=np.array(patch)

        R, G, B = cv2.split(patch)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)

        background_label_img = R * B * G
        forground_label_img = np.ones((1024, 1024)) - background_label_img
        original_result[i:i + 1024, w - 1024:w] = forground_label_img

    for j in range(0, w - 1024, 1024):
        patch = original_img.read_region((j, h - 1024), 0, (1024, 1024)).convert('RGB')
        patch=np.array(patch)
        R, G, B = cv2.split(patch)
        _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
        _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
        _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)

        background_label_img = R * B * G
        forground_label_img = np.ones((1024, 1024)) - background_label_img
        original_result[h - 1024:h, j:j + 1024] = forground_label_img

    e = cv2.resize(original_result, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)

    mask = label(original_result)[0]
    mask = remove_small_objects(mask, 256 * 256, connectivity=5)

    a = (mask > 0).astype(int)

    #t_img = t_img * a

    for i in range(0, h - 1024, 1024):
        for j in range(0, w - 1024, 1024):
            t_img[i:i+1024,j:j+1024]=t_img[i:i+1024,j:j+1024]*a[i:i+1024,j:j+1024]

    for i in range(0, h - 1024, 1024):
        t_img[h-1024:h,j:j+1024]=t_img[h-1024:h,j:j+1024]*a[h-1024:h,j:j+1024]

    for j in range(0, w - 1024, 1024):
        t_img[i:i+1024,w-1024:w]=t_img[i:i+1024,w-1024:w]*a[i:i+1024,w-1024:w]


   # mask = cv2.resize(t_img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)

    #v_img = t_img * v_img
    for i in range(0, h - 1024, 1024):
        for j in range(0, w - 1024, 1024):
            v_img[i:i+1024,j:j+1024]=v_img[i:i+1024,j:j+1024]*t_img[i:i+1024,j:j+1024]

    for i in range(0, h - 1024, 1024):
        v_img[h-1024:h,j:j+1024]=v_img[h-1024:h,j:j+1024]*t_img[h-1024:h,j:j+1024]

    for j in range(0, w - 1024, 1024):
        v_img[i:i+1024,w-1024:w]=v_img[i:i+1024,w-1024:w]*t_img[i:i+1024,w-1024:w]


    #e = cv2.resize(v_img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)

    #plt.imsave('result/' + filename[-7:-4] + '_tumor.png', mask > 0)
    #plt.imsave('result/' + filename[-7:-4] + '_viable.png', e)
    tifffile.imsave('result/' + filename[-7:-4] + '_t.tif', t_img, compress=9)
    tifffile.imsave('result/' + filename[-7:-4] + '.tif', v_img, compress=9)
    print('remove small object')
