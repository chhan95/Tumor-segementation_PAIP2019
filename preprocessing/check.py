import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import glob

file_list=glob.glob('result/*.tif')

for file in file_list:
    print(file)
    img=tifffile.imread(file).astype(np.int8)

    e=cv2.resize(img,(img.shape[0]//40,img.shape[1]//40))
    plt.imsave( file[:-4]+ '.png', e)