import torch
import torch.nn as nn
from .generators import UNetGenerator
from .discriminators import Discriminator
from .losses import CategoryLoss, BinaryLoss
import os
from torch.optim.lr_scheduler import StepLR
from utils.init_net import init_net
import torchvision.utils as vutils
from PIL import Image
import cv2
import numpy as np
from typing import List, Union
import subprocess

class Zi2ZiModel:
    def __init__(self, input_nc=3, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, conv2_layer_count=11, weight_decay = 1e-5, sequence_count=9, final_channels=512):

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.weight_decay = weight_decay    # L2 正則化強度
        self.is_training = is_training
        self.image_size = image_size
        self.conv2_layer_count = conv2_layer_count
        self.sequence_count = sequence_count
        self.final_channels = final_channels

    def setup(self):

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            conv2_layer_count=self.conv2_layer_count
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            sequence_count=self.sequence_count,
            final_channels=self.final_channels,
            image_size=self.image_size
        )

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()
            self.sigmoid.cuda()

        if self.is_training:
            self.netD.train()
            self.netG.train()
        else:
            self.netD.eval()
            self.netG.eval()

    def set_input(self, labels, real_A, real_B):
        if self.gpu_ids:
            self.real_A = real_A.to(self.gpu_ids[0])
            self.real_B = real_B.to(self.gpu_ids[0])
            self.labels = labels.to(self.gpu_ids[0])
        else:
            self.real_A = real_A
            self.real_B = real_B
            self.labels = labels

    def forward(self):
        # generate fake_B
        self.fake_B, self.encoded_real_A = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG(self.fake_B).view(self.fake_B.shape[0], -1)

    def backward_D(self, no_target_source=False):
        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB.detach())

        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty

        d_loss_real = self.real_binary_loss(real_D_logits)
        d_loss_fake = self.fake_binary_loss(fake_D_logits)

        self.d_loss = d_loss_real + d_loss_fake + category_loss / 2.0
        self.d_loss.backward()
        return category_loss

    def backward_G(self, no_target_source=False):

        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        # encoding constant loss
        # this loss assume that generated imaged and real image should reside in the same space and close to each other
        const_loss = self.Lconst_penalty * self.mse(self.encoded_real_A, self.encoded_fake_B)
        # L1 loss between real and generated images
        l1_loss = self.L1_penalty * self.l1_loss(self.fake_B, self.real_B)
        fake_category_loss = self.Lcategory_penalty * self.category_loss(fake_category_logits, self.labels)

        cheat_loss = self.real_binary_loss(fake_D_logits)

        self.g_loss = cheat_loss + l1_loss + fake_category_loss + const_loss
        self.g_loss.backward()
        return const_loss, l1_loss, cheat_loss

    def update_lr(self):
        # There should be only one param_group.
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0001)
            p['lr'] = update_lr
            print("Decay net_D learning rate from %.5f to %.5f." % (current_lr, update_lr))

        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            update_lr = current_lr / 2.0
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0001)
            p['lr'] = update_lr
            print("Decay net_G learning rate from %.5f to %.5f." % (current_lr, update_lr))

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        category_loss = self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights

        # magic move to Optimize G again
        # according to https://github.com/carpedm20/DCGAN-tensorflow
        # collect all the losses along the way
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        const_loss, l1_loss, cheat_loss = self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # udpate G's weights
        return const_loss, l1_loss, category_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in ['G', 'D']:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def save_networks(self, epoch):
        """Save all the networks to the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    # torch.save(net.cpu().state_dict(), save_path)
                    torch.save(net.state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def load_networks(self, epoch):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G', 'D']:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                save_dir_path = os.path.abspath(self.save_dir)
                load_path = os.path.join(save_dir_path, load_filename)
                net = getattr(self, 'net' + name)

                if self.gpu_ids and torch.cuda.is_available():
                    net.load_state_dict(torch.load(load_path, weights_only=True))
                else:
                    net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu'), weights_only=True))
                # net.eval()
        print('load model %d' % epoch)

    def save_image(self, tensor: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        grid = vutils.make_grid(tensor)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        #im = Image.fromarray(ndarr)
        return ndarr

    def anti_aliasing(self, image, strength=1):
        ksize = max(1, strength * 2 + 1)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred

    def sample(self, batch, basename, src_char_list=None, crop_src_font=False, canvas_size=256, resize_canvas_size=256, filename_mode="seq", binary_image=True, strength = 0, image_ext="png"):
        #chk_mkdir(basename)
        cnt = 0
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()

            label_dir = ""
            tensor_to_plot = torch.cat([self.fake_B, self.real_B], 3)
            for label, image_tensor in zip(batch[0], tensor_to_plot):
                label_dir = os.path.join(basename, str(label.item()))

                image_filename = str(cnt)
                if filename_mode != "seq":
                    if src_char_list:
                        if len(src_char_list) > cnt:
                            if filename_mode == "char":
                                image_filename = src_char_list[cnt]
                            if filename_mode == "unicode_hex":
                                image_filename = str(hex(ord(src_char_list[cnt])))
                                if len(image_filename) > 0:
                                    image_filename = image_filename[2:]
                            if filename_mode == "unicode_int":
                                image_filename = str(ord(src_char_list[cnt]))
                saved_image_path = os.path.join(label_dir, image_filename + '.' + image_ext)
                if image_ext == "svg":
                    saved_image_path = os.path.join(label_dir, image_filename + '.pgm')
                
                #vutils.save_image(image_tensor, saved_image_path)
                opencv_image = self.save_image(image_tensor)
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

                if crop_src_font:
                    croped_image = opencv_image[0:canvas_size, 0:canvas_size]
                    if resize_canvas_size > 0 and canvas_size != resize_canvas_size:
                        croped_image = cv2.resize(croped_image, (resize_canvas_size, resize_canvas_size), interpolation=cv2.INTER_LINEAR)
                    else:
                        croped_image = cv2.resize(croped_image, (canvas_size * 2, canvas_size * 2), interpolation=cv2.INTER_LINEAR)
                        croped_image = self.anti_aliasing(croped_image, 1)
                        croped_image = cv2.resize(croped_image, (canvas_size, canvas_size), interpolation=cv2.INTER_LINEAR)
                    croped_image = self.anti_aliasing(croped_image, strength)
                    opencv_image = croped_image
                if binary_image:
                    threshold = 127
                    ret, opencv_image = cv2.threshold(opencv_image, threshold, 255, cv2.THRESH_BINARY)
                
                cv2.imwrite(saved_image_path, opencv_image)
                if image_ext == "svg":
                    saved_svg_path = os.path.join(label_dir, image_filename + '.svg')
                    shell_cmd = 'potrace -b svg -u 60 %s -o %s' % (saved_image_path, saved_svg_path)
                    returned_value = subprocess.call(shell_cmd, shell=True)
                    os.remove(saved_image_path)

                cnt += 1


def chk_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)