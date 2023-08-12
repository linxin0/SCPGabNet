import math
import numbers
import torch
from torch import nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import time
from skimage import io

class GaussionSmoothLayer(nn.Module):
    def __init__(self, channel, kernel_size, sigma, dim = 2):
        super(GaussionSmoothLayer, self).__init__()
        kernel_x = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_y = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel_x * kernel_y.T
        self.kernel_data = kernel
        self.groups = channel
        if dim == 1:
            self.conv = nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 2: 
            self.conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
        elif dim == 3:
            self.conv = nn.Conv3d(in_channels=channel, out_channels=channel, kernel_size=kernel_size, \
                                    groups= channel, bias= False)
            raise RuntimeError(
                'input dim is not supported !, please check it !'
            )
        self.conv.weight.requires_grad = False
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(kernel))
        self.pad = int((kernel_size - 1) / 2)
    def forward(self, input):
        intdata = input
        intdata = F.pad(intdata, (self.pad, self.pad, self.pad, self.pad), mode='reflect')

        output = self.conv(intdata)
        return output

class LapLasGradient(nn.Module):
    def __init__(self, indim, outdim):
        super(LapLasGradient, self).__init__()
        # @ define the sobel filter for x and y axis 
        kernel = torch.tensor(
            [[0, -1, 0],
             [-1, 4, -1],
             [0, -1, 0]
            ]
        )
        kernel2 = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]
            ]
        )
        kernel3 = torch.tensor(
            [[-1, -1, -1],
             [-1, 8, -1],
             [-1, -1, -1]
            ]
        )
        kernel4 = torch.tensor(
            [[1, 1, 1],
             [1, -8, 1],
             [1, 1, 1]
            ]
        )
        self.conv = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv.weight.data.copy_(kernel4)
        self.conv.weight.requires_grad = False

    def forward(self, x):
        grad = self.conv(x)
        return grad



class GradientLoss(nn.Module):
    def __init__(self, indim, outdim):
        super(GradientLoss, self).__init__()
        # @ define the sobel filter for x and y axis 
        x_kernel = torch.tensor(
            [ [1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]
            ]
        )
        y_kernel = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]
            ]
        )
        self.conv_x = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_y = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)

        self.conv_x.weight.data.copy_(x_kernel)
        self.conv_y.weight.data.copy_(y_kernel)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        gradient = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return gradient
class GradientLoss_v1(nn.Module):
    def __init__(self, indim, outdim):
        super(GradientLoss_v1, self).__init__()
        # @ define the sobel filter for x and y axis
        x_kernel = torch.tensor(
            [[0, -1, 0],
             [0, 0, 0],
             [0, 1, 0]]
        )
        y_kernel = torch.tensor(
            [[0, 0, 0],
             [-1, 0, 1],
             [0, 0, 0]]
        )
        self.conv_x = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)
        self.conv_y = nn.Conv2d(indim, outdim, 3, 1, padding= 1, bias=False)

        self.conv_x.weight.data.copy_(x_kernel)
        self.conv_y.weight.data.copy_(y_kernel)
        self.conv_x.weight.requires_grad = False
        self.conv_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        gradient = torch.sqrt(torch.pow(grad_x, 2) + torch.pow(grad_y, 2))
        return gradient
def Norm(x):
    Max_item = torch.max(x)
    Min_item = torch.min(x)
    return (x-Min_item)/(Max_item-Min_item)

class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False)
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x







def main(file1,file2):
    mat = cv2.imread(file1)
    nmat = cv2.imread(file2)
    tensor = torch.from_numpy(mat).float()
    tensor1 = torch.from_numpy(nmat).float()

    # 11, 17, 25, 50 
    blurkernel = GaussionSmoothLayer(3, 11, 50)
    gradloss = GradientLoss(3, 3)
    

    tensor = tensor.permute(2, 0, 1)
    tensor = torch.unsqueeze(tensor, dim = 0)
    
    tensor1 = tensor1.permute(2, 0, 1)
    tensor1 = torch.unsqueeze(tensor1, dim = 0)
    
    out = blurkernel(tensor)
    out1 = blurkernel(tensor1)

    loss = gradloss(out)
    loss1 = gradloss(out1)

    out = out.permute(0, 2, 3, 1).int()
    out = out.numpy().squeeze().astype(np.uint8)

    out1 = out1.permute(0, 2, 3, 1).int()
    out1 = out1.numpy().squeeze().astype(np.uint8)

    cv2.imshow("1", out)
    cv2.imshow("2", out1)
    cv2.waitKey(0)

#   \
#                                          
def testPIL(file1, file2):
    transform = transforms.Compose([
                                    transforms.ToTensor()
                                    ])
    image11 = transform(Image.open(file1).convert('RGB')).unsqueeze(0)
    image22 = transform(Image.open(file2).convert('RGB')).unsqueeze(0)
    # image1 = image11 - F.avg_pool2d(
    #     F.pad(image11, (2, 2, 2, 2), mode='reflect'), 5, 1, padding=0)

    blurkernel = GaussionSmoothLayer(3, 15, 25)
    # blurkernel = Get_gradient()
    # blurkerne2 = GradientLoss_v1(3,3)
    # image1 = image11-blurkernel(image11)
    image2 = blurkernel(image11)


    # # print('自定义花费时间为：{:.8f}s'.format(time.time() - t1))
    # image1 = Norm(image1)
    # image2 = Norm(image2)
    # print(F.mse_loss(image11, image22+image1))


    image2 = image2.numpy().squeeze()

    #
    #
    #
    image2 = np.transpose(image2, (1, 2, 0))

    io.imsave('NOI_SRGB_%d_%d.png' % (1, 1), np.uint8(np.round(image2*255)))


    # image2 = ((image22+image1).clamp_(0,1)).numpy().squeeze()

    # image2 = np.transpose(image2, (1,2,0))
    #
    # img3 = np.transpose(img3, (1, 2, 0))



    # plt.figure('1')
    # plt.imshow(img_as_ubyte(image11), interpolation='nearest')
    # plt.figure('2')
    # plt.imshow(img_as_ubyte(image2), interpolation='nearest')
    #
    # plt.figure('3')
    # # plt.imshow(img_as_ubyte(img3), interpolation='nearest')
    # plt.show()

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.0001 * (0.5 ** ((epoch) //40))
    # lr = opt['lr']
    return str(lr)
if __name__ == "__main__":
    # for i in range(0,400):
    #     print("第{:0>3d}epoch的学习率是：".format(i)+adjust_learning_rate(i))
    file2 = './figs/NOISY_SRGB_0_0.png'
    file1 = './figs/NOISY_SRGB_0_0.png'
    # # main(file1, file2)
    testPIL(file1, file2)
    
    
