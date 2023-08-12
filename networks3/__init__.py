#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-01 20:56:15

from networks3.Discriminator import _NetD, DiscriminatorLinear, Discriminator1,\
    VGGStyleDiscriminator128,NLayerDiscriminator
from networks3.UNetG import UNetG, sample_generator,sample_generator_1
from networks3.UNetD import UNetD,DnCNN
from networks3.util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
from functools import partial
from importlib import import_module
import os
# from GaussianSmoothLayer import GaussionSmoothLayer
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torch
from torch import nn
import torch.nn.functional as F
# from .SubBlocks import conv3x3, conv_down
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-03-20 19:48:14
# Adapted from https://github.com/jvanvugt/pytorch-unet
# from .util import pixel_shuffle_down_sampling, pixel_shuffle_up_sampling
import torch
from torch import nn
import torch.nn.functional as F
from networks3.SubBlocks import conv3x3, conv_down
torch_ver = torch.__version__[:3]
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import PixelShuffle, PixelUnshuffle
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

# __all__ = ['PAM_Module', 'CAM_Module']




class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out




class UNetD(nn.Module):
    def __init__(self, in_chn, wf=64, depth=5, relu_slope=0.2):
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

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope, inplace=True))
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(out_size, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * out_size, out_size, kernel_size=1)

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x):
        out = self.block(x)
        # sa_branch = self.SA(out)
        # ca_branch = self.CA(out)
        # res = torch.cat([sa_branch, ca_branch], dim=1)
        # out = self.conv1x1(res)
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
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(out_size, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * out_size, out_size, kernel_size=1)

    def forward(self, x, bridge):
        up = self.up(x)
        # sa_branch = self.SA(up)
        # ca_branch = self.CA(up)
        # res = torch.cat([sa_branch, ca_branch], dim=1)
        # out = self.conv1x1(res)
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
    def __init__(self,n_feat = 64, reduction = 8):
        super(_Residual_Block, self).__init__()
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)  ## Channel Attention
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        # self.in1 = nn.BatchNorm2d(64, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, )
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1)
    #  self.in2 = nn.BatchNorm2d(64, affine=True)

    def forward(self, x):
        identity_data = x
        output = self.relu((self.conv1(x)))

        output = ((self.conv2(output)))
        # sa_branch = self.SA(output)
        # ca_branch = self.CA(output)
        # res = torch.cat([sa_branch, ca_branch], dim=1)
        # res = self.conv1x1(res)

        output = torch.add(self.relu(output), identity_data)
        # output = torch.add(output,identity_data)


        return output

class _NetG_DOWN(nn.Module):
    def __init__(self, stride=2):
        super(_NetG_DOWN, self).__init__()
        # self.Gas = GaussionSmoothLayer(3, 15, 9)
        self.Gas = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, stride=1, padding=3)
        # self.tiza = tizao(3)
        # self.conv_input = nn.Sequential(
        #     nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=1, padding=3, ),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1, ),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=64, kernel_size=stride + 2, stride=stride, padding=1, ),
        #     nn.LeakyReLU(0.2, inplace=True),
        # )
        self.conv_input = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, stride=1, padding=3, ),
        )
        # self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.residual = self.make_layer(_Residual_Block, 6)
        # self.dab = self.make_layer(DAB, 6)
        self.conv_output = nn.Sequential(
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
             nn.LeakyReLU(0.2, inplace=True),
             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, ),
             nn.LeakyReLU(0.2, inplace=True),
              nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1, padding=3, ),
         )
        self.scale = nn.Parameter(torch.randn(3,1,1),requires_grad=True)


    def make_layer(self, block, num_of_layer):

        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, y):

        # z = torch.cat([x, self.scale*y], dim=1)

        # z = torch.cat([x, y], dim=1)
        #  = torch.cat([x, y - self.tiza(y)], dim=1)
        z = torch.cat([x, y - self.Gas(y)], dim=1)


        out = self.conv_input(z)


        out = self.residual(out)

        out = self.conv_output(out)

        return out + x



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
model_class_dict = {}

def regist_model(model_class):
    model_name = model_class.__name__.lower()
    assert not model_name in model_class_dict, 'there is already registered model: %s in model_class_dict.' % model_name
    model_class_dict[model_name] = model_class

    return model_class

def get_model_class(model_name:str):
    model_name = model_name.lower()
    return model_class_dict[model_name]

# import all python files in model folder


class APBSN(nn.Module):
    '''
    Asymmetric PD Blind-Spot Network (AP-BSN)
    '''

    def __init__(self, pd_a=5, pd_b=2, pd_pad=2, R3=True, R3_T=8, R3_p=0.16,
                 bsn='DBSNl', in_ch=3, bsn_base_ch=128, bsn_num_module=9):
        '''
        Args:
            pd_a           : 'PD stride factor' during training
            pd_b           : 'PD stride factor' during inference
            pd_pad         : pad size between sub-images by PD process
            R3             : flag of 'Random Replacing Refinement'
            R3_T           : number of masks for R3
            R3_p           : probability of R3
            bsn            : blind-spot network type
            in_ch          : number of input image channel
            bsn_base_ch    : number of bsn base channel
            bsn_num_module : number of module
        '''
        super().__init__()

        # network hyper-parameters
        self.pd_a = pd_a
        self.pd_b = pd_b
        self.pd_pad = pd_pad
        self.R3 = R3
        self.R3_T = R3_T
        self.R3_p = R3_p

        # define network
        if bsn == 'DBSNl':
            self.bsn = DBSNl(in_ch, in_ch, bsn_base_ch, bsn_num_module)
        else:
            raise NotImplementedError('bsn %s is not implemented' % bsn)

    def forward(self, img, pd=None):
        '''
        Foward function includes sequence of PD, BSN and inverse PD processes.
        Note that denoise() function is used during inference time (for differenct pd factor and R3).
        '''
        # default pd factor is training factor (a)
        if pd is None: pd = self.pd_a

        # do PD
        if pd > 1:
            pd_img = pixel_shuffle_down_sampling(img, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            pd_img = F.pad(img, (p, p, p, p))

        # forward blind-spot network
        pd_img_denoised = self.bsn(pd_img)

        # do inverse PD
        if pd > 1:
            img_pd_bsn = pixel_shuffle_up_sampling(pd_img_denoised, f=pd, pad=self.pd_pad)
        else:
            p = self.pd_pad
            img_pd_bsn = pd_img_denoised[:, :, p:-p, p:-p]

        return img_pd_bsn

    def denoise(self, x):
        '''
        Denoising process for inference.
        '''
        b, c, h, w = x.shape

        # pad images for PD process
        if h % self.pd_b != 0:
            x = F.pad(x, (0, 0, 0, self.pd_b - h % self.pd_b), mode='constant', value=0)
        if w % self.pd_b != 0:
            x = F.pad(x, (0, self.pd_b - w % self.pd_b, 0, 0), mode='constant', value=0)

        # forward PD-BSN process with inference pd factor
        img_pd_bsn = self.forward(img=x, pd=self.pd_b)

        # Random Replacing Refinement
        if not self.R3:
            ''' Directly return the result (w/o R3) '''
            return img_pd_bsn[:, :, :h, :w]
        else:
            denoised = torch.empty(*(x.shape), self.R3_T, device=x.device)
            for t in range(self.R3_T):
                indice = torch.rand_like(x)
                mask = indice < self.R3_p

                tmp_input = torch.clone(img_pd_bsn).detach()
                tmp_input[mask] = x[mask]
                p = self.pd_pad
                tmp_input = F.pad(tmp_input, (p, p, p, p), mode='reflect')
                if self.pd_pad == 0:
                    denoised[..., t] = self.bsn(tmp_input)
                else:
                    denoised[..., t] = self.bsn(tmp_input)[:, :, p:-p, p:-p]

            return torch.mean(denoised, dim=-1)

        '''
        elif self.R3 == 'PD-refinement':
            s = 2
            denoised = torch.empty(*(x.shape), s**2, device=x.device)
            for i in range(s):
                for j in range(s):
                    tmp_input = torch.clone(x_mean).detach()
                    tmp_input[:,:,i::s,j::s] = x[:,:,i::s,j::s]
                    p = self.pd_pad
                    tmp_input = F.pad(tmp_input, (p,p,p,p), mode='reflect')
                    if self.pd_pad == 0:
                        denoised[..., i*s+j] = self.bsn(tmp_input)
                    else:
                        denoised[..., i*s+j] = self.bsn(tmp_input)[:,:,p:-p,p:-p]
            return_denoised = torch.mean(denoised, dim=-1)
        else:
            raise RuntimeError('post-processing type not supported')
        '''


class DBSNl(nn.Module):
    '''
    Dilated Blind-Spot Network (cutomized light version)

    self-implemented version of the network from "Unpaired Learning of Deep Image Denoising (ECCV 2020)"
    and several modificaions are included.
    see our supple for more details.
    '''

    def __init__(self, in_ch=3, out_ch=3, base_ch=96, num_module=8):
        '''
        Args:
            in_ch      : number of input channel
            out_ch     : number of output channel
            base_ch    : number of base channel
            num_module : number of modules in the network
        '''
        super().__init__()

        assert base_ch % 2 == 0, "base channel should be divided with 2"

        ly = []
        ly += [nn.Conv2d(in_ch, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        self.head = nn.Sequential(*ly)

        self.branch1 = DC_branchl(2, base_ch, num_module)
        self.branch2 = DC_branchl(3, base_ch, num_module)
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(out_ch, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * out_ch, out_ch, kernel_size=1)
        ly = []
        ly += [nn.Conv2d(base_ch * 2, base_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(base_ch , out_ch, kernel_size=3, padding=1,bias=True)]
        self.tail = nn.Sequential(*ly)

    def forward(self, x):

        x = self.head(x)

        br1 = self.branch1(x)
        br2 = self.branch2(x)

        x = torch.cat([br1, br2], dim=1)

        return self.tail(x)

    def _initialize_weights(self):
        # Liyong version
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)


class DC_branchl(nn.Module):
    def __init__(self, stride, in_ch, num_module):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]


        ly += [DCl(stride, in_ch) for _ in range(num_module)]

        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]
        ly += [nn.ReLU(inplace=True)]

        self.body = nn.Sequential(*ly)
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(in_ch, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        y = self.body(x)

        return y

class DCl(nn.Module):
    def __init__(self, stride, in_ch):
        super().__init__()

        ly = []
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        ly += [nn.ReLU(inplace=True)]
        ly += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)]
        ly += [nn.ReLU(inplace=True)]
        # ly += [nn.Conv2d(in_ch, in_ch, kernel_size=1)]

        self.body = nn.Sequential(*ly)
        self.SA = spatial_attn_layer()  ## Spatial Attention
        self.CA = CALayer(in_ch, 8)  ## Channel Attention
        self.conv1x1 = nn.Conv2d(2 * in_ch, in_ch, kernel_size=1)
    def forward(self, x):
        y = self.body(x)

        # z = y1 + x
        # y2 = self.body(z)
        # y3 = y2 + z
        return y+x
        # sa_branch = self.SA(y)
        # ca_branch = self.CA(y)
        # res = torch.cat([sa_branch, ca_branch], dim=1)
        # out = self.conv1x1(res)
        return y+x

class CentralMaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

if __name__ == '__main__':
    net=DBSNl()
    para=sum(p.numel() for p in net.parameters())
    print(para)

class ConvLayer1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride= stride)

        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        return self.conv2d(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1)//2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)

    def forward(self, x):
        return self.block(x)


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x ,y ):
        return torch.mul((1-self.delta), x) + torch.mul(self.delta, y)


class Encoding_block(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Encoding_block, self).__init__()
        self.n_convblock = n_convblock
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        for i in range(self.n_convblock-1):
            x = self.body[i](x)
        ecode = x
        x = self.body[self.n_convblock-1](x)
        return ecode, x


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.conv2d(x_in)
        return out


class upsample1(nn.Module):
    def __init__(self, base_filter):
        super(upsample1, self).__init__()
        self.conv1 = ConvLayer(base_filter, base_filter, 3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(base_filter, base_filter, kernel_size=3, stride=1, upsample=2)
        self.cat = ConvLayer1(base_filter*2, base_filter, kernel_size=1, stride=1)

    def forward(self, x, y):
        y = self.ConvTranspose(y)
        x = self.conv1(x)
        return self.cat(torch.cat((x, y), dim=1))


class Decoding_block2(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Decoding_block2, self).__init__()
        self.n_convblock = n_convblock
        self.upsample = upsample1(base_filter)
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y):
        x = self.upsample(x, y)
        for i in range(self.n_convblock):
            x = self.body[i](x)
        return x

#Corresponds to DEAM Module in NLO Sub-network
class Attention_unet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attention_unet, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )
        self.cat = ConvLayer1(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(channel, channel, kernel_size=3, stride=1, upsample=2)#up-sampling

    def forward(self, x, g):
        up_g = self.ConvTranspose(g)
        weight = self.conv_du(self.cat(torch.cat([self.C(x), up_g], 1)))
        rich_x = torch.mul((1 - weight), up_g) + torch.mul(weight, x)
        return rich_x

#Corresponds to NLO Sub-network
class ziwangluo1(nn.Module):
    def __init__(self, base_filter, n_convblock_in, n_convblock_out):
        super(ziwangluo1, self).__init__()
        self.conv_dila1 = ConvLayer1(64, 64, 3, 1)
        self.conv_dila2 = ConvLayer1(64, 64, 5, 1)
        self.conv_dila3 = ConvLayer1(64, 64, 7, 1)

        self.cat1 = torch.nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, stride=1, padding=0,
                                    dilation=1, bias=True)
        nn.init.xavier_normal_(self.cat1.weight.data)
        self.e3 = Encoding_block(base_filter, n_convblock_in)
        self.e2 = Encoding_block(base_filter, n_convblock_in)
        self.e1 = Encoding_block(base_filter, n_convblock_in)
        self.e0 = Encoding_block(base_filter, n_convblock_in)


        self.attention3 = Attention_unet(base_filter)
        self.attention2 = Attention_unet(base_filter)
        self.attention1 = Attention_unet(base_filter)
        self.attention0 = Attention_unet(base_filter)
        self.mid = nn.Sequential(ConvLayer(base_filter, base_filter, 3, 1),
                                 ConvLayer(base_filter, base_filter, 3, 1))
        self.de3 = Decoding_block2(base_filter, n_convblock_out)
        self.de2 = Decoding_block2(base_filter, n_convblock_out)
        self.de1 = Decoding_block2(base_filter, n_convblock_out)
        self.de0 = Decoding_block2(base_filter, n_convblock_out)

        self.final = ConvLayer1(base_filter, base_filter, 3, stride=1)

    def forward(self, x):
        _input = x
        encode0, down0 = self.e0(x)
        encode1, down1 = self.e1(down0)
        encode2, down2 = self.e2(down1)
        encode3, down3 = self.e3(down2)

        # media_end = self.Encoding_block_end(down3)
        media_end = self.mid(down3)

        g_conv3 = self.attention3(encode3, media_end)
        up3 = self.de3(g_conv3, media_end)
        g_conv2 = self.attention2(encode2, up3)
        up2 = self.de2(g_conv2, up3)

        g_conv1 = self.attention1(encode1, up2)
        up1 = self.de1(g_conv1, up2)

        g_conv0 = self.attention0(encode0, up1)
        up0 = self.de0(g_conv0, up1)

        final = self.final(up0)

        return _input+final


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x, y):
        return torch.mul((1 - self.delta), x) + torch.mul(self.delta, y)


class SCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y


class Weight(nn.Module):
    def __init__(self, channel):
        super(Weight, self).__init__()
        self.cat =ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)

    def forward(self, x, y):
        delta = self.weight(self.cat(torch.cat([self.C(y), x], 1)))
        return delta


class transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),

        )

    def forward(self, x):
        y = self.ext(x)
        return y+self.pre(y)


class Inverse_transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inverse_transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.pre(x)+x
        x = self.ext(x)
        return x


class Deam(nn.Module):
    def __init__(self, Isreal):
        super(Deam, self).__init__()
        if Isreal:
            self.transform_function = transform_function(3, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 3)
        else:
            self.transform_function = transform_function(1, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 1)

        self.line11 = Weight(64)
        self.line22 = Weight(64)
        self.line33 = Weight(64)
        self.line44 = Weight(64)

        self.net2 = ziwangluo1(64, 3, 2)

    def forward(self, x):
        x = self.transform_function(x)
        y = x

        # Corresponds to NLO Sub-network
        x1 = self.net2(y)
        # Corresponds to DEAM Module
        delta_1 = self.line11(x1, y)
        x1 = torch.mul((1 - delta_1), x1) + torch.mul(delta_1, y)

        x2 = self.net2(x1)
        delta_2 = self.line22(x2, y)
        x2 = torch.mul((1 - delta_2), x2) + torch.mul(delta_2, y)

        x3 = self.net2(x2)
        delta_3 = self.line33(x3, y)
        x3 = torch.mul((1 - delta_3), x3) + torch.mul(delta_3, y)

        x4 = self.net2(x3)
        delta_4 = self.line44(x4, y)
        x4 = torch.mul((1 - delta_4), x4) + torch.mul(delta_4, y)
        x4 = self.inverse_transform_function(x4)
        return x4


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
