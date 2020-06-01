
import matplotlib.pyplot as plt
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import chkpts as cp
from config import Config

class cpBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(cpBatchNorm2d, self).__init__(*args, **kwargs)

    def forward(self, input):
        self._check_input_dim(input)
        if input.requires_grad:
            exponential_average_factor = 0.0
            if self.training and self.track_running_stats:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / self.num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats,
                exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var, self.weight, self.bias,
                self.training or not self.track_running_stats, 0.0, self.eps)

class Net(nn.Module):
    """ 
    A base class provides a common weight initialization scheme.
    """

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if 'conv' in classname.lower():
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            if 'norm' in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if 'linear' in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    @staticmethod
    def save_mem(prev_feat, blk_func):
        if any(feat.requires_grad for feat in prev_feat):
            args = (prev_feat,) + tuple(blk_func.parameters()) 
            feat = cp.CheckpointFunction.apply(blk_func, 1, *args)
        else:
            feat = blk_func(prev_feat)
        return feat
        
    def forward(self, x):
        return x

class ResUnit(Net, Config):
    def __init__(self, in_ch, ksize, ch, stride=1, split=1, efficient=True):
        super(ResUnit, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.efficient = efficient
        # NOTE: may be more complicated if dilate etc. involve
        pad = [kernel // 2 for kernel in ksize]
        
        self.norm1 = nn.GroupNorm(in_ch // 4, in_ch, eps=1e-5)    
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, ch[0], ksize[0], stride=1, padding=pad[0], bias=False)

        self.norm2 = nn.GroupNorm(ch[0] // 4, ch[0], eps=1e-5)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch[0], ch[1], ksize[1], stride=stride, padding=pad[1], 
                                                        groups=split, bias=False)

        self.norm3 = nn.GroupNorm(ch[1] // 4, ch[1], eps=1e-5)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(ch[1], ch[2], ksize[2], stride=1, padding=pad[2], bias=False)

        self.shorcut = None
        if in_ch != ch[-1] or stride != 1:
            self.shorcut = nn.Sequential(OrderedDict([
                ('norm' , nn.GroupNorm(in_ch // 4, in_ch, eps=1e-5)), 
                ('relu' , nn.ReLU(inplace=True)),
                ('conv' , nn.Conv2d(in_ch, ch[-1], 1, stride=stride))
            ]))

    def forward(self, prev_feat):
        if self.efficient and any(feat.requires_grad for feat in prev_feat):
            args = (prev_feat,) + tuple(self.norm1.parameters()) + tuple(self.conv1.parameters())
            func = lambda x: self.conv1(self.relu1(self.norm1(x)))
            feat = cp.CheckpointFunction.apply(func, 1, *args)
        else:
            feat = self.conv1(self.relu1(self.norm1(prev_feat)))
       
        feat = self.conv2(self.relu2(self.norm2(feat)))
        feat = self.conv3(self.relu3(self.norm3(feat)))

        shortcut = prev_feat if self.shorcut is None else self.shorcut(prev_feat)
        return feat + shortcut

class ResBlock(Net, Config):
    def __init__(self, in_ch, ksize, ch, nr_unit, split=1, stride=1):
        super(ResBlock, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.nr_unit = nr_unit

        # wrapper so that params in list are visible
        self.res_units = nn.ModuleList() 
       
        self.res_units.append(ResUnit(in_ch, ksize, ch, stride, split))
        for i in range (1, nr_unit):
            self.res_units.append(ResUnit(ch[-1], ksize, ch, 1, split))

    def forward(self, prev_feat):
        for idx in range (0, self.nr_unit):
            unit_blk = self.res_units[idx]
            prev_feat = unit_blk(prev_feat)
        return prev_feat

class DenseUnit(Net, Config):   
    def __init__(self, in_ch, ksize, ch, padding='same', efficient=True):
        super(DenseUnit, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.efficient = efficient
        self.padding = padding
        # NOTE: may be more complicated if dilate etc. involve
        if padding == 'same':
            pad = [kernel // 2 for kernel in ksize]
        else:
            pad = [0 for kernel in ksize] 

        self.norm1 = nn.GroupNorm(in_ch // 4, in_ch, eps=1e-5)    
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, ch[0], ksize[0], stride=1, padding=pad[0], bias=False)

        self.norm2 = nn.GroupNorm(ch[0] // 4, ch[0], eps=1e-5)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch[0], ch[1], ksize[1], stride=1, padding=pad[1], bias=False)

    def forward(self, prev_feat):
        if self.efficient and any(feat.requires_grad for feat in prev_feat):
            args = (prev_feat,) + tuple(self.norm1.parameters()) + tuple(self.conv1.parameters())
            func = lambda x: self.conv1(self.relu1(self.norm1(x)))
            feat = cp.CheckpointFunction.apply(func, 1, *args)
        else:
            feat = self.conv1(self.relu1(self.norm1(prev_feat)))
       
        feat = self.conv2(self.relu2(self.norm2(feat)))
        feat = torch.cat([prev_feat, feat], dim=1)
        return feat
        
class DenseBlock(Net, Config):
    def __init__(self, in_ch, out_ch, ksize, ch, nr_unit, padding='same'):
        super(DenseBlock, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.nr_unit = nr_unit
        # define 1 dense unit
        unit_input_ch = in_ch

        # wrapper so that params in list are visible
        self.dense_units = nn.ModuleList() 
        for i in range (0, nr_unit):
            self.dense_units.append(DenseUnit(unit_input_ch, ksize, ch, padding))
            unit_input_ch += ch[1]

        self.blk_final = nn.Sequential(
            nn.GroupNorm(unit_input_ch // 4, unit_input_ch, eps=1e-5), 
            nn.ReLU(inplace=True),
            nn.Conv2d(unit_input_ch, out_ch, 5, stride=1, padding=2)
        )

    def forward(self, prev_feat):
        for idx in range (0, self.nr_unit):
            dense_unit = self.dense_units[idx]
            prev_feat = dense_unit(prev_feat)       
        feat = self.save_mem(prev_feat, self.blk_final)
        return feat

class DenseNet(Net, Config):
    def __init__(self, input_ch, nr_classes):
        super(DenseNet, self).__init__()
        Config.__init__(self) # TODO: why need this redundancy here?

        self.d0 = nn.Sequential(
            nn.Conv2d(input_ch, 48, 9, stride=1, padding=4),
        )

        self.d1 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            DenseBlock(48, 64, [5, 5], [32, 16], 4)
        )

        self.d2 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            DenseBlock(64, 96, [5, 5], [32, 16], 8)
        )

        self.d3 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            DenseBlock(96, 144, [5, 5], [32, 16], 12)
        )

        self.d4 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            DenseBlock(144, 264, [5, 5], [32, 16], 24)
        )

        self.d5 = nn.Sequential(
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            DenseBlock(264, 264, [5, 5], [32, 16], 24)
        )

        self.u4 = DenseBlock(264, 144, [5, 5], [32, 16], 8)
        self.u3 = DenseBlock(144,  96, [5, 5], [32, 16], 6)
        self.u2 = DenseBlock( 96,  64, [5, 5], [32, 16], 4)
        self.u1 = DenseBlock( 64,  48, [5, 5], [32, 16], 4)

        self.conv_out = nn.Sequential(
            nn.GroupNorm(48 // 4, 48, eps=1e-5), 
            nn.ReLU(), 
            nn.Conv2d(48, nr_classes, 1),
        )
        self.softmax=nn.LogSoftmax(dim=1)
        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()

    def forward(self, imgs):
        ####
        def crop_op(x, cropping, data_format='NCHW'):
            """
            Center crop image
            Args:
                cropping is the substracted portion
            """
            crop_t = cropping[0] // 2
            crop_b = cropping[0] - crop_t
            crop_l = cropping[1] // 2
            crop_r = cropping[1] - crop_l
            if data_format == 'NCHW':
                x = x[:,:,crop_t:-crop_b,crop_l:-crop_r]
            else:
                x = x[:,crop_t:-crop_b,crop_l:-crop_r]
            return x  
        ####
        def upsample(x):
            return nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        ####

        imgs = imgs / 255.0 # to 0-1 range to match XY

        d0 = self.d0(imgs)
        d1 = self.d1(d0)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)

        u4 = upsample(d5)
        u4 = self.u4(u4 + d4)

        u3 = upsample(u4)
        u3 = self.u3(u3 + d3)

        u2 = upsample(u3)
        u2 = self.u2(u2 + d2)
        
        u1 = upsample(u2)
        u1 = self.u1(u1 + d1)
        
        u0 = upsample(u1)
        u0 = u0 + d0
        
        # out = self.save_mem(u0, self.conv_out)
        out = self.conv_out(u0)

        return out
