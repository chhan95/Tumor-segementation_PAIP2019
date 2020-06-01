import os
import glob
import shutil
import argparse

import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import misc
import dataset
from net import DenseNet
from config import Config

def rm_n_mkdir(dir):
    if (os.path.isdir(dir)):
        shutil.rmtree(dir)
    os.makedirs(dir) 

class Inferer(Config):
    def infer_step(self, net, batch):
        net.eval() # infer mode

        imgs = torch.FloatTensor(batch) # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs) # forward
            prob = nn.functional.softmax(logit, dim=1)
            prob = prob.permute(0, 2, 3, 1) # to NHWC
            return prob.cpu().numpy()

    def run(self):
        def center_pad_to(img, h, w):
            shape = img.shape

            diff_h = h - shape[0]
            padt = diff_h // 2
            padb = diff_h - padt

            diff_w = w - shape[1]
            padl = diff_w // 2
            padr = diff_w - padl

            img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), 
                            'constant', constant_values=255)
            return img

        input_chs = 3    
        net = DenseNet(input_chs, self.nr_classes)

        saved_state = torch.load(self.inf_model_path)
        pretrained_dict = saved_state.module.state_dict() # due to torch.nn.DataParallel        
        net.load_state_dict(pretrained_dict, strict=False)
        net = net.to('cuda')

        file_list = glob.glob('%s/*%s' % (self.inf_imgs_dir, self.inf_imgs_ext))
        file_list.sort() # ensure same order

        if not os.path.isdir(self.inf_output_dir):
            os.makedirs(self.inf_output_dir) 
        
        cmap = plt.get_cmap('jet')
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            print(filename, ' ---- ', end='', flush=True)

            img = cv2.imread(self.inf_imgs_dir + filename)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
            img = cv2.resize(img, (0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

            orig_shape = img.shape
            img = center_pad_to(img, 2880, 2880)
            pred = self.infer_step(net, [img])[0,...,1:]
            pred = misc.cropping_center(pred, orig_shape[:-1])

            # plt.subplot(1,3,1)
            # plt.imshow(img)
            # plt.subplot(1,3,2)
            # plt.imshow(pred[...,0])
            # plt.subplot(1,3,3)
            # plt.imshow(pred[...,1])
            # plt.show()
            # exit()            
            np.save('%s/%s.npy' % (self.inf_output_dir, basename), pred)

            # epi = cmap(pred[0,...,2])[...,:3] # gray to RGB heatmap
            # epi = (epi * 255).astype('uint8')
            # epi = cv2.cvtColor(epi, cv2.COLOR_RGB2BGR)

            # cv2.imwrite('%s/%s.png' % (self.inf_output_dir, basename), epi)
            print('FINISH')
####

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    inferer = Inferer()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu       
    nr_gpus = len(args.gpu.split(','))
    inferer.run()