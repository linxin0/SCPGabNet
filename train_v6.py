import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as uData
from networks3 import _NetG_DOWN,NLayerDiscriminator,Deam
from datasets.DenoisingDatasets import BenchmarkTrain, BenchmarkTest
from math import ceil
from utils import *
from loss import get_gausskernel, GANLoss, log_SSIM_loss
import warnings
from pathlib import Path
import commentjson as json
from GaussianSmoothLayer import GaussionSmoothLayer
# filter warnings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0'if torch.cuda.is_available()else'cpu')
warnings.simplefilter('ignore', Warning, lineno=0)
torch.set_default_dtype(torch.float32)
_C = 3
_modes = ['train', 'val']
BGBlur_kernel = [3, 9, 15]
BlurWeight = [0.01,0.1,1.]
# For blurring of BGMLOSS
BlurNet = [GaussionSmoothLayer(3, k_size, 25).cuda() for k_size in BGBlur_kernel]

def main():
    with open('./configs/DANet_v5.json', 'r') as f:
        args = json.load(f)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu_id"]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # build up the E when SC
    netE = torch.nn.DataParallel(Deam(1)).cuda()
    # build up the denoiser
    print('start')
    netD = torch.nn.DataParallel(Deam(1)).cuda()
    print('for')
    # build up the generator
    netG = torch.nn.DataParallel(_NetG_DOWN(stride=1)).cuda()
    # build up the discriminator
    netP = torch.nn.DataParallel(NLayerDiscriminator(6)).cuda()


    criterionGAN = GANLoss(args['gan_mode']).cuda()
    init_weights(netG, init_type='normal',init_gain=0.02)
    init_weights(netP, init_type='normal', init_gain=0.02)
    # !!!Before the SC, No optimizerE and netE
    net = {'E':netE,'D': netD, 'G': netG, 'P': netP}
    # optimizer
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr_G'])
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr_D'])
    optimizerP = optim.Adam(netP.parameters(), lr=args['lr_P'])
    optimizer = {'D': optimizerD, 'G': optimizerG, 'P': optimizerP}
    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            # args['epoch_start'] = 3
            # optimizerE.load_state_dict(checkpoint['optimizer_state_dict']['E'])
            # optimizerD.load_state_dict(checkpoint['optimizer_state_dict']['D'])
            # optimizerG.load_state_dict(checkpoint['optimizer_state_dict']['G'])
            # optimizerP.load_state_dict(checkpoint['optimizer_state_dict']['P'])
            netE.load_state_dict(checkpoint['model_state_dict']['D'])
            netD.load_state_dict(checkpoint['model_state_dict']['D'])
            netG.load_state_dict(checkpoint['model_state_dict']['G'])
            netP.load_state_dict(checkpoint['model_state_dict']['P'])
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if not Path(args['log_dir']).is_dir():
            Path(args['log_dir']).mkdir()
        if not Path(args['model_dir']).is_dir():
            Path(args['model_dir']).mkdir()

    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))

    # making dataset, out dataset are hdf5
    datasets = {'train': BenchmarkTrain(h5_file=args['SIDD_train_h5_noisy'],
                                        length=2000 * args['batch_size'] * args['num_critic'],
                                        pch_size=args['patch_size'],
                                        mask=False),
                'val': BenchmarkTest(args['SIDD_test_h5'])}

    # build the Gaussian kernel for loss
    global kernel
    kernel = get_gausskernel(args['ksize'], chn=_C).cuda()
    # train model
    print('\nBegin training with GPU: ' + (args['gpu_id']))
    train_epoch(net, datasets, optimizer, args, criterionGAN)

def train_epoch(net, datasets, optimizer, args, criterionGAN):
    criterion = nn.L1Loss().cuda()
    loss_ssim = log_SSIM_loss().cuda()
    batch_size = {'train': args['batch_size'], 'val': 4}
    data_loader = {phase: uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                           shuffle=True, num_workers=0, pin_memory=True) for phase in
                   _modes}
    data_set_gt = BenchmarkTrain(h5_file=args['SIDD_train_h5_gt'],
                                 length=2000 * args['batch_size'] * args['num_critic'],
                                 pch_size=args['patch_size'],
                                 mask=False)
    # todo gt dataset has no key()
    data_loader_gt = uData.DataLoader(data_set_gt, batch_size=batch_size['train'],
                                      shuffle=True, num_workers=0, pin_memory=True)

    num_data = {phase: len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}

    for epoch in range(args['epoch_start'], args['epochs']):
        loss_epoch = {x: 0 for x in ['PL', 'DL', 'GL']}
        subloss_epoch = {x: 0 for x in
                         ['loss_GAN_DG', 'loss_l1', 'perceptual_loss', 'loss_bgm', 'loss_GAN_P_real', 'loss_GAN_P_fake']}
        mae_epoch = {'train': 0, 'val': 0}

        optD, optP, optG =  optimizer['D'],optimizer['P'], optimizer['G']


        tic = time.time()
        # train stage
        net['D'].train()
        net['G'].train()
        net['P'].train()

        lr_D = optimizer['D'].param_groups[0]['lr']
        lr_G = optimizer['G'].param_groups[0]['lr']
        lr_P = optimizer['P'].param_groups[0]['lr']

        if lr_D < 1e-6:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'

        for ii, (data, data1) in enumerate(zip(data_loader[phase], data_loader_gt)):

            im_noisy,im_gt = [x.cuda() for x in data]
            ################################
            #training generator
            ##############################
            optimizer['G'].zero_grad()
            optimizer['D'].zero_grad()
            # !!!first stage, No SC
            # fake_im_noisy1 = net['G'](im_gt, im_noisy)
            # rec_x1 = net['D'](fake_im_noisy1.detach())
            # rec_x2 = net['D'](im_noisy.detach())
            # fake_im_noisy2 = net['G'](rec_x2, im_noisy)
            # fake_im_noisy3 = net['G'](rec_x2, fake_im_noisy1)
            # fake_im_noisy4 = net['G'](rec_x1, fake_im_noisy1)


            # SC stage
            tizao_1 = net['E'](im_noisy)
            rec_x2 = net['D'](im_noisy.detach())
            fake_im_noisy1 = net['G'](im_gt, (im_noisy-tizao_1))
            fake_im_noisy2 = net['G'](rec_x2, (im_noisy - tizao_1))
            rec_x1 = net['D'](fake_im_noisy1.detach())
            tizao_2 = net['E'](fake_im_noisy1)
            fake_im_noisy3 = net['G'](rec_x2, (fake_im_noisy1 - tizao_2))
            fake_im_noisy4 = net['G'](rec_x1, (fake_im_noisy1 - tizao_2))


            set_requires_grad([net['P']], False)

            subloss_epoch['perceptual_loss'] += 0
            adversarial_loss1 = criterionGAN(net['P'](fake_im_noisy1), True)
            adversarial_loss2 = criterionGAN(net['P'](fake_im_noisy2), True)
            adversarial_loss3 = criterionGAN(net['P'](fake_im_noisy3), True)
            adversarial_loss4 = criterionGAN(net['P'](fake_im_noisy4), True)

            adversarial_loss = adversarial_loss1+adversarial_loss2+adversarial_loss3+adversarial_loss4
            identity_loss = 0
            bgm_loss1 = 0
            bgm_loss2 = 0
            bgm_loss  = 0
# 求BCM
            for index, weight in enumerate(BlurWeight):
                out_b1 = BlurNet[index](im_gt)
                out_real_b1 = BlurNet[index](fake_im_noisy1)
                out_b2 = BlurNet[index](rec_x2)
                out_real_b2 = BlurNet[index](fake_im_noisy2)
                grad_loss_b1 = criterion(out_b1, out_real_b1)
                grad_loss_b2 = criterion(out_b2, out_real_b2)
                bgm_loss1 += weight * (grad_loss_b1)
                bgm_loss2 += weight * (grad_loss_b2)
                bgm_loss  += bgm_loss1 + bgm_loss2
            loss_G =  adversarial_loss * args['adversarial_loss_factor'] + \
                     bgm_loss1 * args['bgm_loss'] + \
                     bgm_loss2 * args['bgm_loss']


            los_ssim = loss_ssim(rec_x1, im_gt)
            loss_recon = criterion(rec_x1, im_gt)
            # first stage no SC
            # loss_D = loss_recon + los_ssim

            # SC stage
            los_ssim1 = loss_ssim(rec_x2, tizao_1)
            loss_recon1 = criterion(rec_x2, tizao_1)
            los_ssim2 = loss_ssim(rec_x1, tizao_2)
            loss_recon2 = criterion(rec_x1, tizao_2)
            loss_D = loss_recon + los_ssim + loss_recon2 + los_ssim2 + los_ssim1 + loss_recon1


            loss_G.backward(retain_graph=True)

            loss_D.backward(retain_graph=True)
            optimizer['G'].step()
            optimizer['D'].step()
            loss_epoch['DL'] += loss_D.item()
            loss_epoch['GL'] += loss_G.item()

            subloss_epoch['loss_GAN_DG'] += adversarial_loss.item()
            subloss_epoch['loss_bgm'] += bgm_loss.item()

            ##########################
            # training discriminator #
            ##########################
            if (ii+1) % args['num_critic'] == 0:
                set_requires_grad([net['P']], True)

                pred_real1 = net['P'](im_noisy)
                loss_P_real = criterionGAN(pred_real1, True)
                pred_fake = net['P'](fake_im_noisy1.detach())
                loss_P_fake = criterionGAN(pred_fake, False)

                # Combined loss and calculate gradients
                loss_P = (loss_P_real + loss_P_fake) * 0.5
                loss_P.backward()
                optimizer['P'].step()
                optimizer['P'].zero_grad()

                loss_epoch['PL'] += loss_P.item()
                subloss_epoch['loss_GAN_P_real'] += loss_P_real.item()
                subloss_epoch['loss_GAN_P_real'] += loss_P_fake.item()

                if (ii + 1) % args['print_freq'] == 0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}/{:0>5d},' + \
                                   ' PL:{:>6.6f}, GL:{:>6.6f}, DL:{:>6.6f}, ' \
                                   'loss_GAN_G:{:>6.6f},' + \
                                   'loss_bgm:{:>6.9f}, loss_P_real:{:>6.4f}, ' \
                                   'loss_P_fake:{:>6.4f}, indentity_loss:{:>6.4f}'
                    print(template.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch[phase],
                                              loss_P.item(), loss_G.item(),loss_D.item(),
                                              # loss_P1.item(), loss_G.item(), loss_D.item(),
                                              adversarial_loss.item(), bgm_loss1.item(), loss_P_real.item(), loss_P_fake.item(),identity_loss))

        loss_epoch['GL'] /= (ii + 1)

        subloss_epoch['loss_GAN_DG'] /= (ii + 1)
        subloss_epoch['loss_bgm'] /= (ii + 1)
        subloss_epoch
        loss_epoch['PL'] /= (ii + 1)
        subloss_epoch['loss_GAN_P_real'] /= (ii + 1)
        subloss_epoch['loss_GAN_P_fake'] /= (ii + 1)

        template = '{:s}: PL:{:>6.6f}, GL:{:>6.6f},loss_GAN_DG:{:>6.6f}, ' + \
                   ' loss_bgm:{:>6.4f}, loss_P_real:{:>6.4f}, ' \
                   'loss_P_fake:{:>6.4f}, lrDG/P:{:.2e}/{:.2e}'
        print(template.format(phase, loss_epoch['PL'], loss_epoch['GL'], subloss_epoch['loss_GAN_DG'],
                              subloss_epoch['loss_bgm'],
                              subloss_epoch['loss_GAN_P_real'],
                              subloss_epoch['loss_GAN_P_fake'], lr_D, lr_P))

        net['G'].eval()
        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'
            .format(
            epoch,
            lr=lr_D,
            loss=loss_epoch['DL']))

        print('-' * 150)

        # test stage
        net['D'].eval()
        psnr_per_epoch = ssim_per_epoch = 0
        phase = 'val'
        for ii, data in enumerate(data_loader[phase]):
            im_noisy, im_gt = [x.cuda() for x in data]
            with torch.set_grad_enabled(False):
                im_denoise = net['D'](im_noisy)

            mae_iter = F.l1_loss(im_denoise, im_gt)
            im_denoise.clamp_(0.0, 1.0)
            mae_epoch[phase] += mae_iter
            psnr_iter = batch_PSNR(im_denoise, im_gt)
            psnr_per_epoch += psnr_iter
            ssim_iter = batch_SSIM(im_denoise, im_gt)
            ssim_per_epoch += ssim_iter
            if (ii + 1) % 50 == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}, ' + \
                          'psnr={:4.2f}, ssim={:5.4f}'
                print(log_str.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch[phase],
                                     mae_iter, psnr_iter, ssim_iter))

        psnr_per_epoch /= (ii + 1)
        ssim_per_epoch /= (ii + 1)
        mae_epoch[phase] /= (ii + 1)
        print('{:s}: mae={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'.format(phase, mae_epoch[phase],
                                                                    psnr_per_epoch, ssim_per_epoch))
        print('-' * 150)

        # save model
        model_prefix = 'model_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1)))
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': {x: net[x].state_dict() for x in ['E','D',  'G', 'P']},
            'optimizer_state_dict': {x: optimizer[x].state_dict() for x in ['D', 'P', 'G']},
        }, save_path_model)
        model_prefix = 'model_state_'
        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1) + 'PSNR{:.2f}_SSIM{:.4f}'.
                                                         format(psnr_per_epoch, ssim_per_epoch) + '.pt'))
        torch.save({x: net[x].state_dict() for x in ['E','D', 'G', 'P']}, save_path_model)

        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))

    print('Reach the maximal epochs! Finish training')

# 不更新梯度，判别器中使用
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or no
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def adjust_learning_rate(epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt['lr'] * (opt['gamma'] ** ((epoch) // opt['lr_decay']))
    # lr = opt['lr']
    return lr


if __name__ == '__main__':
    main()
