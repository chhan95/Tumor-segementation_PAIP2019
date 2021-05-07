import tifffile
import torch
from torch_scatter import scatter_max

import glob
import matplotlib.pyplot as plt
import openslide
import numpy as np
# img=openslide.OpenSlide("dataset/2019-03-29 11.41.41.ndpi")

# a=(img.read_region((2000,2000),0,(2000,2000))).convert("RGB")
a=tifffile.imread("/media/workspace/han/tumorsegmentation_paip2019/output/2019-03-29 11_wt.tif")
a=np.array(a)
plt.imshow(a)
plt.show()