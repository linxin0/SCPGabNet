#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-01-22 22:07:08

import math
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from loss import get_gausskernel, gaussblur
import numpy as np
import cv2

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_psnr(im1, im2, border=0):
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = im1.shape[:2]
    im1 = im1[border:h-border, border:w-border]
    im2 = im2[border:h-border, border:w-border]

    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    mse = np.mean((im1 - im2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def batch_PSNR(img, imclean, border=0):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += calculate_psnr(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (PSNR/Img.shape[0])

def batch_SSIM(img, imclean, border=0):
    Img = img.data.cpu().numpy()
    Iclean = imclean.data.cpu().numpy()
    Img = img_as_ubyte(Img)
    Iclean = img_as_ubyte(Iclean)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += calculate_ssim(Iclean[i,:,].transpose((1,2,0)), Img[i,:,].transpose((1,2,0)), border)
    return (SSIM/Img.shape[0])

def kl_gauss_zero_center(sigma_fake, sigma_real):
    '''
    Input:
        sigma_fake: 1 x C x H x W, torch array
        sigma_real: 1 x C x H x W, torch array
    '''
    div_sigma = torch.div(sigma_fake, sigma_real)
    div_sigma.clamp_(min=0.1, max=10)
    log_sigma = torch.log(1 / div_sigma)
    distance = 0.5 * torch.mean(log_sigma + div_sigma - 1.)
    return distance

def estimate_sigma_gauss(img_noisy, img_gt):
    win_size = 7
    err2 = (img_noisy - img_gt) ** 2
    kernel = get_gausskernel(win_size, chn=3).to(img_gt.device)
    sigma = gaussblur(err2, kernel, win_size, chn=3)
    sigma.clamp_(min=1e-10)

    return sigma

class PadUNet:
    '''
    im: N x C x H x W torch tensor
    dep_U: depth of UNet
    '''
    def __init__(self, im, dep_U, mode='reflect'):
        self.im_old = im
        self.dep_U = dep_U
        self.mode = mode
        self.H_old = im.shape[2]
        self.W_old = im.shape[3]

    def pad(self):
        lenU = 2 ** (self.dep_U-1)
        padH = 0 if ((self.H_old % lenU) == 0) else (lenU - (self.H_old % lenU))
        padW = 0 if ((self.W_old % lenU) == 0) else (lenU - (self.W_old % lenU))
        padding = (0, padW, 0, padH)
        out = F.pad(self.im_old, pad=padding, mode=self.mode)
        return out

    def pad_inverse(self, im_new):
        return im_new[:, :, :self.H_old, :self.W_old]

from torch.nn import init
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

