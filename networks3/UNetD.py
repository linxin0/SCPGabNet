#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet

import torch
from torch import nn
import torch.nn.functional as F
from .SubBlocks import conv3x3, conv_down, conv1x1


class UNetD(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2):
        super(UNetD, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        # self._initialize()

    def forward(self, x1):
        res = x1
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        out = self.last(x1)
        return out+res

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class BasicConv(nn.Module):
        def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                     bn=False, bias=False):
            super(BasicConv, self).__init__()
            self.out_channels = out_planes
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                  dilation=dilation, groups=groups, bias=bias)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
            self.relu = nn.ReLU() if relu else None

        def forward(self, x):
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
            if self.relu is not None:
                x = self.relu(x)
            return x
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class CALayer(nn.Module):
        def __init__(self, channel, reduction=16):
            super(CALayer, self).__init__()
            # global average pooling: feature --> point
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            # feature channel downscale and upscale --> channel weight
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv_du(y)
            return x * y
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(in_size, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * in_size, out_size, kernel_size=1)


    def forward(self, x):
        out = self.block(x)
        sa_branch = self.SA(out)
        ca_branch = self.CA(out)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        out = self.conv1x1(res)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out

class DnCNN(nn.Module):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
    def forward(self, x):
        out = self.dncnn(x)
        return out

class tizao(nn.Module):
    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2):
        super(tizao, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        prev_channels = self.get_input_chn(in_chn)
        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            prev_channels = (2**i)*wf

        self.last = conv1x1(prev_channels, in_chn, bias=True)
        # self._initialize()

    def forward(self, x1):
        res = x1
        blocks = []
        for i, down in enumerate(self.down_path):
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path):
            x1 = up(x1, blocks[-i-1])

        out = self.last(x1)
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class shengcheng(nn.Module):
    def __init__(self, stride=2):
        super(shengcheng, self).__init__()
    def forward(self, x, y):
        # z = torch.cat([x, self.scale*y], dim=1)

        # z = torch.cat([x, y], dim=1)
        z = torch.cat([x, y - tizao.y], dim=1)
        out = self.conv_input(z)

        out = self.residual(out)

        out = self.conv_output(out)

        return out + x

