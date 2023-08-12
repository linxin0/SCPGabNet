#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import sys
import torch
import h5py as h5
import random
import cv2
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch.utils.data as uData
from skimage import img_as_float32 as img_as_float
# from . import BaseDataSetH5, BaseDataSetFolder
class BaseDataSetH5(uData.Dataset):
    def __init__(self, h5_path, length=None):
        '''
        Args:
            h5_path (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetH5, self).__init__()
        self.h5_path = h5_path
        self.length = length
        with h5.File(h5_path, 'r') as h5_file:
            self.keys = list(h5_file.keys())
            self.num_images = len(self.keys)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, imgs_sets):
        H, W, C2 = imgs_sets.shape
        # minus the bayer patter channel
        C = int(C2/2)
        ind_H = random.randint(0, H-self.pch_size)
        ind_W = random.randint(0, W-self.pch_size)
        im_noisy = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, :C])
        im_gt = np.array(imgs_sets[ind_H:ind_H+self.pch_size, ind_W:ind_W+self.pch_size, C:])
        return im_gt, im_noisy

class BaseDataSetFolder(uData.Dataset):
    def __getitem__(self, path_list, pch_size, length=None):
        '''
        Args:
            path_list (str): path of the hdf5 file
            length (int): length of Datasets
        '''
        super(BaseDataSetFolder, self).__init__()
        self.path_list = path_list
        self.length = length
        self.pch_size = pch_size
        self.num_images = len(path_list)

    def __len__(self):
        if self.length == None:
            return self.num_images
        else:
            return self.length

    def crop_patch(self, im):
        pch_size = self.pch_size
        H, W, _ = im.shape
        if H < self.pch_size or W < self.pch_size:
            H = max(pch_size, H)
            W = max(pch_size, W)
            im = cv2.resize(im, (W, H))
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch = im[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch

def data_augmentation(image, mode):
    '''
    Performs data augmentation of the input image
    Input:
        image: a cv2 (OpenCV) image
        mode: int. Choice of transformation to apply to the image
                0 - no transformation
                1 - flip up and down
                2 - rotate counterwise 90 degree
                3 - rotate 90 degree and flip up and down
                4 - rotate 180 degree
                5 - rotate 180 degree and flip
                6 - rotate 270 degree
                7 - rotate 270 degree and flip
    '''
    if mode == 0:
        # original
        out = image
    elif mode == 1:
        # flip up and down
        out = np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(image)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(image, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(image, k=3)
        out = np.flipud(out)
    else:
        raise Exception('Invalid choice of image transformation')

    return out

def random_augmentation(*args):
    out = []
    if random.randint(0,1) == 1:
        flag_aug = random.randint(1,7)
        for data in args:
            out.append(data_augmentation(data, flag_aug).copy())
    else:
        for data in args:
            out.append(data)
    return out

# Benchmardk Datasets: and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128, mask=False):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.pch_size = pch_size
        self.mask = mask


    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images - 1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.mask:
            return im_noisy, im_gt, torch.ones((1, 1, 1), dtype=torch.float32)
        else:
            return im_noisy, im_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt

class FakeTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128):
        super(FakeTrain, self).__init__(path_list, pch_size, length)

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        im_gt = img_as_float(cv2.imread(self.path_list[ind_im], 1)[:, :, ::-1])
        im_gt = self.crop_patch(im_gt)

        # data augmentation
        im_gt = random_augmentation(im_gt)[0]

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))

        return im_gt, im_gt, torch.zeros((1,1,1), dtype=torch.float32)

class PolyuTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128, mask=False):
        super(PolyuTrain, self).__init__(path_list, pch_size, length)
        self.mask = mask

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        path_noisy = self.path_list[ind_im]
        head, tail = os.path.split(path_noisy)
        path_gt = os.path.join(head, tail.replace('real', 'mean'))
        im_noisy = img_as_float(cv2.imread(path_noisy, 1)[:, :, ::-1])
        im_gt = img_as_float(cv2.imread(path_gt, 1)[:, :, ::-1])
        im_noisy, im_gt = self.crop_patch(im_noisy, im_gt)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.mask:
            return im_noisy, im_gt, torch.ones((1,1,1), dtype=torch.float32)
        else:
            return im_noisy, im_gt

    def crop_patch(self, im_noisy, im_gt):
        pch_size = self.pch_size
        H, W, _ = im_noisy.shape
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch_noisy = im_noisy[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        im_pch_gt = im_gt[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch_noisy, im_pch_gt
