from skimage.filters import try_all_threshold
import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2

from scipy.ndimage import label
from skimage.morphology import remove_small_objects
import tifffile


file_list=glob.glob('basic/*tif')
for filename in file_list:

    if filename[-5]=='t':
        continue
    print(filename)
    img=tifffile.imread(filename)
    img=np.array(img)
    h,w =img.shape


    # R, G, B = cv2.split(patch)
    # _, R = cv2.threshold(R, 235, 1, cv2.THRESH_BINARY)
    # _, B = cv2.threshold(B, 235, 1, cv2.THRESH_BINARY)
    # _, G = cv2.threshold(G, 210, 1, cv2.THRESH_BINARY)
    #
    # background_label_img = R * B * G
    # forground_label_img = np.ones((1024, 1024)) - background_label_img
    mask=label(img)[0]
    mask=remove_small_objects(mask,h*w//8000,connectivity=10)
    img=cv2.resize(img,(0,0),fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)
    mask=cv2.resize(mask,(0,0),fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)

    plt.imsave('result/'+filename[-7:-4]+'_original.png', img)
    plt.imsave('result/'+filename[-7:-4]+'_remove.png', mask > 0)