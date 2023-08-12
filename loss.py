#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-10-31 21:31:50

import torch
import torch.nn.functional as F
import functools
from math import exp
import cv2
import numpy as np
import torch.nn as nn
from torchvision.models.vgg import vgg19
from loss_util import weighted_loss
from GaussianSmoothLayer import GaussionSmoothLayer
_reduction_modes = ['none', 'mean', 'sum']

from torch.autograd import Variable


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

def gradient_penalty(real_data, generated_data, netP, lambda_gp):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, 1, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(real_data.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated.requires_grad=True

        # Calculate probability of interpolated examples
        prob_interpolated = netP(interpolated)

        # Calculate gradients of probabilities with respect to examples
        grad_outputs = torch.ones(prob_interpolated.size(), device=real_data.device, dtype=torch.float32)
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                 grad_outputs=grad_outputs, create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return  lambda_gp * ((gradients_norm - 1) ** 2).mean()

def get_gausskernel(p, chn=3):
    '''
    Build a 2-dimensional Gaussian filter with size p
    '''
    x = cv2.getGaussianKernel(p, sigma=-1)   # p x 1
    y = np.matmul(x, x.T)[np.newaxis, np.newaxis,]  # 1x 1 x p x p
    out = np.tile(y, (chn, 1, 1, 1)) # chn x 1 x p x p

    return torch.from_numpy(out).type(torch.float32)

def gaussblur(x, kernel, p=5, chn=3):
    x_pad = F.pad(x, pad=[int((p-1)/2),]*4, mode='reflect')
    y = F.conv2d(x_pad, kernel, padding=0, stride=1, groups=chn)

    return y

def var_match(x, y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err2_real = (err_real-mu_real)**2
    var_real = gaussblur(err2_real, kernel, p, chn)
    var_real = torch.where(var_real<1e-10, torch.ones_like(fake_y)*1e-10, var_real)
    # estimate the fake distribution
    err_fake = fake_y - x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    err2_fake = (err_fake-mu_fake)**2
    var_fake = gaussblur(err2_fake, kernel, p, chn)
    var_fake = torch.where(var_fake<1e-10, torch.ones_like(fake_y)*1e-10, var_fake)

    # calculate the loss
    loss_err = F.l1_loss(mu_real, mu_fake, reduction='mean')
    loss_var = F.l1_loss(var_real, var_fake, reduction='mean')

    return loss_err, loss_var

def mean_match(x, fake_y,y,fake_x, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    err_real = fake_y - x
    mu_real = gaussblur(err_real, kernel, p, chn)
    err_fake = y - fake_x
    mu_fake = gaussblur(err_fake, kernel, p, chn)
    loss = F.l1_loss(mu_real, mu_fake, reduction='mean')

    return loss

def mean_match_1(y, fake_y, kernel, chn=3):
    p = kernel.shape[2]
    # estimate the real distribution
    # err_real = y - x
    mu_real = gaussblur(y, kernel, p, chn)

    mu_fake = gaussblur(fake_y, kernel, p, chn)
    loss = F.l1_loss(mu_real, mu_fake, reduction='mean')

    return loss

class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GANLoss_v2(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss_v2, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
class GANLoss_v3(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self,
                 gan_type,
                 real_label_val=1.0,
                 fake_label_val=0.0,
                 loss_weight=1.0):
        super(GANLoss_v3, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(
                f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(
            input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (
            self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError('Unsupported reduction mode: {reduction}. '
                             'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.Gas = GaussionSmoothLayer(3, 15, 9)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # pred = pred - F.avg_pool2d(
        #     F.pad(pred, (1, 1, 1, 1), mode='reflect'), 3, 1, padding=0)
        # target = target - F.avg_pool2d(
        #     F.pad(target, (1, 1, 1, 1), mode='reflect'), 3, 1, padding=0)
        pred = pred - self.Gas(pred)
        target = target - self.Gas(target)
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


class log_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(log_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return -torch.log10(ssim_map.mean())


class negative_SSIM_loss(nn.Module):
    def __init__(self, window_size=11, channel=3, is_cuda=True, size_average=True):
        super(negative_SSIM_loss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.size_average = size_average
        self.window = create_window(window_size, channel)
        if is_cuda:
            self.window = self.window.cuda()


    def forward(self, img1, img2):
        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1.0-ssim_map.mean()


class GRAD_loss(nn.Module):
    def __init__(self, channel=3, is_cuda=True):
        super(GRAD_loss, self).__init__()
        self.edge_conv = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = []
        for i in range(channel):
            edge_k.append(edge_kx)
            edge_k.append(edge_ky)

        edge_k = np.stack(edge_k)

        edge_k = torch.from_numpy(edge_k).float().view(channel*2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

        if is_cuda: self.edge_conv.cuda()

    def forward(self, img1, img2):
        img1_grad = self.edge_conv(img1)
        img2_grad = self.edge_conv(img2)

        return F.l1_loss(img1_grad, img2_grad)


class exp_GRAD_loss(nn.Module):
    def __init__(self, channel=3, is_cuda=True):
        super(exp_GRAD_loss, self).__init__()
        self.edge_conv = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1, groups=channel, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = []
        for i in range(channel):
            edge_k.append(edge_kx)
            edge_k.append(edge_ky)

        edge_k = np.stack(edge_k)

        edge_k = torch.from_numpy(edge_k).float().view(channel*2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        for param in self.parameters():
            param.requires_grad = False

        if is_cuda: self.edge_conv.cuda()

    def forward(self, img1, img2):
        img1_grad = self.edge_conv(img1)
        img2_grad = self.edge_conv(img2)

        return torch.exp(F.l1_loss(img1_grad, img2_grad)) - 1


class log_PSNR_loss(torch.nn.Module):
    def __init__(self):
        super(log_PSNR_loss, self).__init__()

    def forward(self, img1, img2):
        diff = img1 - img2
        mse = diff*diff.mean()
        return -torch.log10(1.0-mse)


class MSE_loss(torch.nn.Module):
    def __init__(self):
        super(MSE_loss, self).__init__()

    def forward(self, img1, img2):
        return F.mse_loss(img1, img2)


class L1_loss(torch.nn.Module):
    def __init__(self):
        super(L1_loss, self).__init__()

    def forward(self, img1, img2):
        return F.l1_loss(img1, img2)


loss_dict = {
    'l1': L1_loss,
    'mse': MSE_loss,
    'grad': GRAD_loss,
    'exp_grad': exp_GRAD_loss,
    'log_ssim': log_SSIM_loss,
    'neg_ssim': negative_SSIM_loss,
    'log_psnr': log_PSNR_loss,
}

