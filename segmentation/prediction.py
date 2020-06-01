
import numpy as np
import matplotlib.pyplot as plt
import openslide
import shutil
import argparse
import os
import json
import random
import warnings
from termcolor import colored
import pandas as pd
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.utils.data as data

import imgaug # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa

import misc
import dataset
from net import DenseNet
from config import Config
import cv2

device='cuda'

net = DenseNet(3, 2)
net.eval()  # infer mode

viable_saved_state=torch.load('log/v1.0.0.1/model_net_46.pth')

new_saved_state={}

for key,value in viable_saved_state.items():
    new_saved_state[key[7:]]=value

net.load_state_dict(new_saved_state)
net=torch.nn.DataParallel(net).to(device)


wsi_img=openslide.OpenSlide('01_01_0138.svs')
wsi_w,wsi_h=wsi_img.level_dimensions[0]

prediction=np.zeros((wsi_h,wsi_w))
batch=[]
location=[]
batch_size=80
one=np.ones((512,512))
for i in range(0,wsi_h,512):
    for j in range(0,wsi_w,512):
        print('{0} {1}'.format(i,j))
        # if i+512>wsi_h and j+512>wsi_w:
        #     patch = wsi_img.read_region((wsi_w - 512, wsi_h-512), 0, (512, 512)).convert('RGB')
        #     img = np.array(patch)
        #     batch.append(img)
        #     location.append((wsi_h-512, wsi_w - 512))
        # elif j+512>wsi_w:
        #     patch = wsi_img.read_region((wsi_w-512, wsi_h), 0, (512, 512)).convert('RGB')
        #     img = np.array(patch)
        #     batch.append(img)
        #     location.append((wsi_h, wsi_w-512))
        # elif i+512>wsi_h:
        #     patch = wsi_img.read_region((wsi_w, wsi_h - 512), 0, (512, 512)).convert('RGB')
        #     img = np.array(patch)
        #     batch.append(img)
        #     location.append((wsi_h - 512, wsi_w ))
        # else:
        if i + 512 > wsi_h or j + 512 > wsi_w:
            continue
        patch = wsi_img.read_region((j, i), 0, (512, 512)).convert('RGB')
        img = np.array(patch)
        batch.append(img)
        location.append((i, j))


        if len(batch)==batch_size:
            with torch.no_grad():
                batch=np.array(batch)
                w=batch
                batch = torch.from_numpy(batch)
                batch = batch.permute(0, 3, 1, 2)
                batch = batch.to(device).float()
                logit = net(batch)
                prob = nn.functional.softmax(logit, dim=1)

                pre = torch.argmax(prob, dim=1)

                pre = pre.cpu()
                for k in range(batch_size):
                    prediction[location[k][0]:location[k][0] + 512, location[k][1]:location[k][1] + 512] = pre[k].numpy()

            batch=[]
            location=[]

pre_h,pre_w=prediction.shape
prediction=cv2.resize(prediction,(pre_h//40,pre_w//40))
plt.imshow(prediction)
plt.show()
plt.imsave('prediction.png',prediction)


