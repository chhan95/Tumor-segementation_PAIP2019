import torch
from torch_scatter import scatter_max

import glob
import matplotlib.pyplot as plt
import tifffile
img=tifffile.imread("output/01_01_0083_wt.tif")
print(img.max())
plt.imshow(img)
plt.show()