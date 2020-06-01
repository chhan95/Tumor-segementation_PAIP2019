import matplotlib.pyplot as plt
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
import chkpts as cp
from unet import unet_parts


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


class UNet(Net, Config):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        Config.__init__(self)  # TODO: why need this redundancy here?

        self.inc = unet_parts.inconv(n_channels, 64)
        self.down1 = unet_parts.down(64, 128)
        self.down2 = unet_parts.down(128, 256)
        self.down3 = unet_parts.down(256, 512)
        self.down4 = unet_parts.down(512, 512)
        self.up1 = unet_parts.up(1024, 256)
        self.up2 = unet_parts.up(512, 128)
        self.up3 = unet_parts.up(256, 64)
        self.up4 = unet_parts.up(128, 64)
        self.outc = unet_parts.outconv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
