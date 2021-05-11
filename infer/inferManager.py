import os.path

import cv2

from utils import *
from classification_model.net import DenseCLSNet as c_net
from segmentation_model.net import DenseUNet as s_net
import glob
import numpy as np
from termcolor import cprint
import openslide
import tifffile

import torch.nn as nn
import torch
import torch.nn.functional as F

from termcolor import colored

import matplotlib.pyplot as plt
class InferManager():
    def __init__(self, args):
        self.input_path = args["input_path"]
        self.output_path = args["output_path"]
        self.step_size = int(args["step_size"])
        self.rescale=float(args["rescale"])
        self.c_window_size = 1024
        self.s_window_size = 512
        self.whole_class_net = self.load_weight("pretrained/whole_cls_model_net_26.pth",num_class=4)
        self.viable_class_net = self.load_weight("pretrained/viable_cls_model_net_52.pth")
        self.ambiguous_class_net = self.load_weight("pretrained/ambiguous_whole_tumor_cls_model_net_31.pth")
        self.viable_seg_net = self.load_weight("pretrained/viable_seg_model_net_46.pth",cls=False)

        mkdir_s(os.path.join(self.output_path,"prediction"))

    def run(self):
        task1_result, task2_result = self.task()
        print("task1",task1_result)
        print("task2",task2_result)
        cprint("END", "green")

    def load_weight(self, path, cls=True, num_class=2):
        if cls:
            net = c_net(3, num_class)
        else:
            net = s_net(3, num_class)
        net.eval()
        state = torch.load(path)
        new_state = {}
        for key, value in state.items():
            new_state[key[7:]] = value

        net.load_state_dict(new_state)
        net=net.to('cuda')
        return net
    def generate_thumnail(self,arr,wsi_name,type):
        resized_arr = cv2.resize(arr, (0, 0), fx=0.01, fy=0.01,interpolation=cv2.INTER_NEAREST)
        _,resized_arr=cv2.threshold(resized_arr,0,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(self.output_path,"thumbnail","{0}_{1}.png".format(wsi_name,type)),resized_arr)
        return
    def task(self):

        # PREDICTION START
        print(colored('PREDICTION START', 'green'))

        wsi_list = glob.glob(self.input_path+"/*.svs")
        wsi_list += glob.glob(self.input_path+"/*.ndpi")

        result_wt_list = []
        result_v_list = []
        with torch.no_grad():
            for wsi_path in wsi_list:
                wsi_name = wsi_path.split("/")[-1]
                wsi_name = wsi_name.split(".")[0]

                cprint(wsi_path, "red")

                wsi_info = openslide.OpenSlide(wsi_path)
                wsi_w, wsi_h = wsi_info.level_dimensions[0]

                # task1 result
                whole_tumor_result = np.zeros((wsi_h, wsi_w, 2))
                viable_tumor_result = np.zeros((wsi_h, wsi_w, 2))

                # sliding window
                for step in range(0,self.c_window_size,self.step_size):
                    whole_tumor_result,viable_tumor_result=self.sliding_window(whole_tumor_result,viable_tumor_result, wsi_info, step)

                whole_tumor_result,viable_tumor_result=self.sliding_window_right(whole_tumor_result,viable_tumor_result,wsi_info,self.step_size)
                whole_tumor_result,viable_tumor_result=self.sliding_window_bottom(whole_tumor_result,viable_tumor_result,wsi_info,self.step_size)

                whole_tumor_result = np.argmax(whole_tumor_result, axis=-1)
                viable_tumor_result = np.argmax(viable_tumor_result, axis=-1)

                viable_tumor_result=viable_tumor_result*whole_tumor_result

                whole_tumor_result = whole_tumor_result.astype('uint8')
                tifffile.imsave(os.path.join(self.output_path,"prediction","{0}_wt.tif".format(wsi_name)) , whole_tumor_result, compress=9)

                viable_tumor_result = viable_tumor_result.astype('uint8')
                tifffile.imsave(os.path.join(self.output_path,"prediction","{0}_v.tif".format(wsi_name)) , viable_tumor_result, compress=9)

                result_wt_list.append([os.path.join(self.output_path,"prediction","{0}_wt.tif".format(wsi_name))])
                result_v_list.append([os.path.join(self.output_path,"prediction","{0}_v.tif".format(wsi_name))])
                
                if self.rescale >0 or self.rescale < 1:
                    mkdir_s(os.path.join(self.output_path, "thumbnail"))
                    self.generate_thumnail(whole_tumor_result,wsi_name,"wt")
                    self.generate_thumnail(viable_tumor_result,wsi_name,"v")

            return sorted(result_wt_list),sorted(result_v_list)

    def sliding_window(self,whole_tumor_result,viable_tumor_result,wsi_info,start_point):
        for x, y, window in get_window(wsi_info, start_point, self.c_window_size,):
            tumor_window = False

            background, foregorund = get_threshold_result(window)

            if foregorund.sum() < 5000:
                continue

            batch = torch.from_numpy(window)
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.permute(0, 3, 1, 2)  # to NCHW
            batch = batch.to('cuda').float()

            # whole_tumor_net
            logit = self.whole_class_net(batch)
            whole_prob = F.softmax(logit, dim=-1)
            whole_pre = torch.argmax(whole_prob, dim=-1)
            whole_prob = whole_prob.cpu()
            whole_pre = whole_pre.cpu()

            # ambiguous_whole_tumor_net
            logit = self.ambiguous_class_net(batch)
            amb_prob = F.softmax(logit, dim=-1)
            amb_pre = torch.argmax(amb_prob, dim=-1)

            logit = self.viable_class_net(batch)
            via_prob = F.softmax(logit, dim=-1)
            via_pre = torch.argmax(via_prob, dim=-1)
            ###
            if 0.2 <= whole_prob[0][1] and whole_prob[0][1] <= 0.8:
                if amb_pre[0] == 1:
                    tumor_window = True
            else:
                if whole_pre[0] == 1:
                    tumor_window = True

            if tumor_window:

                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 0] \
                    += whole_prob[0][0].numpy() * background
                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 1] \
                    += whole_prob[0][1].numpy() * foregorund
                if via_pre[0]==1:
                    viable_tumor_result = seg_prediction(viable_tumor_result, self.viable_seg_net, window, x, y,
                                                     foregorund, window_size=self.s_window_size)
        return whole_tumor_result,viable_tumor_result
    def sliding_window_right(self,whole_tumor_result,viable_tumor_result,wsi_info,step_size):
        for x, y, window in get_window_right(wsi_info, self.c_window_size,step_size):
            tumor_window = False

            background, foregorund = get_threshold_result(window)

            if foregorund.sum() < 5000:
                continue
            batch = torch.from_numpy(window)
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.permute(0, 3, 1, 2)  # to NCHW
            batch = batch.to('cuda').float()

            # whole_tumor_net
            logit = self.whole_class_net(batch)
            whole_prob = F.softmax(logit, dim=-1)
            whole_pre = torch.argmax(whole_prob, dim=-1)
            whole_prob = whole_prob.cpu()
            whole_pre = whole_pre.cpu()

            # ambiguous_whole_tumor_net
            logit = self.ambiguous_class_net(batch)
            amb_prob = F.softmax(logit, dim=-1)
            amb_pre = torch.argmax(amb_prob, dim=-1)
            ###

            logit = self.viable_class_net(batch)
            via_prob = F.softmax(logit, dim=-1)
            via_pre = torch.argmax(via_prob, dim=-1)
            if 0.2 <= whole_prob[0][1] and whole_prob[0][1] <= 0.8:
                if amb_pre[0] == 1:
                    tumor_window = True
            else:
                if whole_pre[0] == 1:
                    tumor_window = True

            if tumor_window:
                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 0] \
                    += whole_prob[0][0].numpy() * background
                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 1] \
                    += whole_prob[0][1].numpy() * foregorund
                if via_pre[0]==1:

                    viable_tumor_result = seg_prediction(viable_tumor_result, self.viable_seg_net, window, x, y,
                                                     foregorund, window_size=self.s_window_size)
        return whole_tumor_result,viable_tumor_result
    def sliding_window_bottom(self, whole_tumor_result, viable_tumor_result, wsi_info,step_size):
        for x, y, window in get_window_bottom(wsi_info, self.c_window_size,step_size):
            tumor_window = False

            background, foregorund = get_threshold_result(window)

            if foregorund.sum() < 5000:
                continue
            batch = torch.from_numpy(window)
            batch = torch.unsqueeze(batch, dim=0)
            batch = batch.permute(0, 3, 1, 2)  # to NCHW
            batch = batch.to('cuda').float()

            # whole_tumor_net
            logit = self.whole_class_net(batch)
            whole_prob = F.softmax(logit, dim=-1)
            whole_pre = torch.argmax(whole_prob, dim=-1)
            whole_prob = whole_prob.cpu()
            whole_pre = whole_pre.cpu()

            # ambiguous_whole_tumor_net
            logit = self.ambiguous_class_net(batch)
            amb_prob = F.softmax(logit, dim=-1)
            amb_pre = torch.argmax(amb_prob, dim=-1)
            ###

            logit = self.viable_class_net(batch)
            via_prob = F.softmax(logit, dim=-1)
            via_pre = torch.argmax(via_prob, dim=-1)
            if 0.2 <= whole_prob[0][1] and whole_prob[0][1] <= 0.8:
                if amb_pre[0] == 1:
                    tumor_window = True
            else:
                if whole_pre[0] == 1:
                    tumor_window = True
            if tumor_window:
                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 0] \
                    += whole_prob[0][0].numpy() * background
                whole_tumor_result[y:y + self.c_window_size, x:x + self.c_window_size, 1] \
                    += whole_prob[0][1].numpy() * foregorund
                if via_pre[0]==1:

                    viable_tumor_result = seg_prediction(viable_tumor_result, self.viable_seg_net, window, x, y,
                                                     foregorund, window_size=self.s_window_size)
        return whole_tumor_result, viable_tumor_result
