#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F
from .SubBlocks import conv3x3, conv_down
from .UNetD import UNetD
from torch.nn import init
class UNetG(UNetD):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.20):
        """
        Reference:
        Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical
        Image Segmentation. MICCAI 2015.
        ArXiv Version: https://arxiv.org/abs/1505.04597

        Args:
            in_chn (int): number of input channels, Default 3
            depth (int): depth of the network, Default 4
            wf (int): number of filters in the first layer, Default 32
        """
        super(UNetG, self).__init__(in_chn, wf, depth, relu_slope)

    def get_input_chn(self, in_chn):
        return in_chn+3



def sample_generator(netG, x):
    z = torch.randn([x.shape[0], 3, x.shape[2], x.shape[3]], device=x.device)
    # x1 = torch.cat([x, z], dim=1)
    out = netG(x,z)

    return out+x

def sample_generator_1(netG, x,z):
    # z = torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]], device=x.device)
    x1 = torch.cat([x, z], dim=1)
    out = netG(x1)

    return out+x


class _Conv_Block(nn.Module):
    def __init__(self):
        super(_Conv_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.in1(self.conv1(x)))
        output = self.in2(self.conv2(output))
        return output


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, )

    #  self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))

        output = self.relu((self.conv2(output)))
        output = torch.add(output, identity_data)
        return output

# class _NetG_DOWN(nn.Module):
#     def __init__(self, stride=2):
#         super(_NetG_DOWN, self).__init__()
#
#         self.conv_input = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, stride=1, padding=3, ),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1, ),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1, ),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
#
#         self.relu = nn.LeakyReLU(0.2, inplace=True)
#
#         self.residual = self.make_layer(_Residual_Block, 6)
#
#         self.conv_output = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, ),
#         )
#
#     def make_layer(self, block, num_of_layer):
#         layers = []
#         for _ in range(num_of_layer):
#             layers.append(block())
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv_input(x)
#
#
#         out = self.residual(out)
#
#
#         out = self.conv_output(out)
#
#         return out

