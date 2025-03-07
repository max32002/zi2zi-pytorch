import functools
import math
import os
import subprocess
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.nn import init
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR

from utils.init_net import init_net


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, self_attention=False, residual_block=False, blur=False):
        super(UNetGenerator, self).__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, layer=1, embedding_dim=embedding_dim, self_attention=self_attention, residual_block=residual_block, blur=blur)
        for index in range(num_downs - 5):  # add intermediate layers with ngf * 8 filtersv
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=index+2, use_dropout=use_dropout, self_attention=self_attention, residual_block=residual_block, blur=blur)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=5, self_attention=self_attention, residual_block=residual_block, blur=blur)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=6, self_attention=self_attention, residual_block=residual_block, blur=blur)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, layer=7, self_attention=self_attention, residual_block=residual_block, blur=blur)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, norm_layer=norm_layer, layer=8, self_attention=self_attention, residual_block=residual_block, blur=blur)
        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x, style_or_label=None):
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            style = self.embedder(style_or_label)
            output = self.model(x, style)
            return output
        else:
            output = self.model(x, style_or_label)
            return output

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, embedding_dim=128, norm_layer=nn.BatchNorm2d, layer=0,
                 use_dropout=False, self_attention=False, residual_block=False, blur=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.attn4 = None
        self.attn6 = None
        if self_attention:
            self.attn4 = SelfAttention(512)
            self.attn6 = SelfAttention(128)
        self.res_block3 = None
        self.res_block5 = None
        if residual_block:
            self.res_block3 = ResidualBlock(512, 512) # 第 3 層的輸出通道數為 512
            self.res_block5 = ResidualBlock(256, 256) # 第 5 層的輸出通道數為 256

        outermost=False
        innermost=False
        if layer==1:
            innermost=True
        if layer==8:
            outermost=True
        self.outermost = outermost
        self.innermost = innermost
        self.layer = layer
        self.self_attention = self_attention
        self.residual_block = residual_block
        self.embedding_dim = embedding_dim
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                up = up + [nn.Dropout(0.5)]

        self.submodule = submodule
        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

        self.blur = blur
        self.gaussian_blur = T.GaussianBlur(kernel_size=1, sigma=1.0)

    def forward(self, x, style=None):
        if self.innermost:
            encode = self.down(x)
            if style is None:
                if self.blur:
                    encode = self.gaussian_blur(encode)
                return encode
            up_input = None
            if self.embedding_dim > 0:
                new_style = style.view(style.shape[0], self.embedding_dim, 1, 1)
                if encode.shape[2] != new_style.shape[2]:
                    new_style = nn.functional.interpolate(new_style, size=[encode.size(2), encode.size(3)], mode='bilinear', align_corners=False)
                up_input = torch.cat([new_style, encode], 1)
            else:
                up_input = encode
            dec = self.up(up_input)
            dec_resized = dec
            if x.shape[2] != dec.shape[2]:
                dec_resized = nn.functional.interpolate(dec, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
            ret1=torch.cat([x, dec_resized], 1)
            ret2=encode.view(x.shape[0], -1)
            if self.blur:
                ret1 = self.gaussian_blur(ret1)
            return ret1, ret2

        elif self.outermost:
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            up_input, encode = self.submodule(enc, style)

            dec = self.up(up_input)
            return dec, encode
        else:  # add skip connections
            enc = self.down(x)
            if style is None:
                return self.submodule(enc)
            up_input, encode = self.submodule(enc, style)
            dec = self.up(up_input)
            if self.self_attention:
                if self.layer == 4:
                    dec = self.attn4(dec) # 加入 self-attention 層
                if self.layer == 6:
                    dec = self.attn6(dec) # 加入 self-attention 層

            if self.residual_block:
                if self.layer == 3:
                    dec = self.res_block3(dec) # 在第 3 層之後加入殘差塊
                if self.layer == 5:
                    dec = self.res_block5(dec) # 在第 5 層之後加入殘差塊

            dec_resized = dec
            if x.shape[2] != dec.shape[2]:
                dec_resized = nn.functional.interpolate(dec, size=[x.size(2), x.size(3)], mode='bilinear', align_corners=False)
            ret1=torch.cat([x, dec_resized], 1)

            return ret1, encode

# 自注意力機制
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        attention = torch.bmm(query, key)
        attention = nn.functional.softmax(attention, dim=-1)
        attention = torch.bmm(value, attention.permute(0, 2, 1))
        attention = attention.view(batch_size, channels, height, width)

        return x + self.gamma * attention

# 殘差塊
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc, embedding_num, ndf=64, norm_layer=nn.BatchNorm2d, image_size=256, sequence_count=9, final_channels=1, blur=False):
        super(Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d
        kw = 5
        padw = 2
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, 3):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = 8
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, final_channels, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(final_channels),
            nn.LeakyReLU(0.2, True)
        ]
        if sequence_count > 8:
            sequence += [nn.Conv2d(final_channels, 1, kernel_size=kw, stride=1, padding=padw)]
            final_channels = 1

        self.model = nn.Sequential(*sequence)
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        image_size = math.ceil(image_size / 2)
        final_features = final_channels * image_size * image_size
        self.binary = nn.Linear(final_features, 1)
        self.catagory = nn.Linear(final_features, embedding_num)
        self.blur = blur
        self.gaussian_blur = T.GaussianBlur(kernel_size=1, sigma=1.0)  # 設定模糊程度

    def forward(self, input):
        features = self.model(input)
        if self.blur:
            features = self.gaussian_blur(features)
        features = features.view(input.shape[0], -1)
        binary_logits = self.binary(features)
        catagory_logits = self.catagory(features)
        return binary_logits, catagory_logits

class CategoryLoss(nn.Module):
    def __init__(self, category_num):
        super(CategoryLoss, self).__init__()
        emb = nn.Embedding(category_num, category_num)
        emb.weight.data = torch.eye(category_num)
        self.emb = emb
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, category_logits, labels):
        target = self.emb(labels)
        return self.loss(category_logits, target)

class BinaryLoss(nn.Module):
    def __init__(self, real):
        super(BinaryLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.real = real

    def forward(self, logits):
        if self.real:
            labels = torch.ones(logits.shape[0], 1)
        else:
            labels = torch.zeros(logits.shape[0], 1)
        if logits.is_cuda:
            labels = labels.cuda()
        return self.bce(logits, labels)


class Zi2ZiModel:
    def __init__(self, input_nc=3, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, self_attention=False, residual_block=False, 
                 weight_decay = 1e-5, sequence_count=9, final_channels=1, new_final_channels=0, beta1=0.5, g_blur=False, d_blur=False):

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
        self.beta1 = beta1
        self.weight_decay = weight_decay    # L2 正則化強度
        self.is_training = is_training
        self.image_size = image_size
        self.self_attention=self_attention
        self.residual_block=residual_block
        self.sequence_count = sequence_count
        self.final_channels = final_channels
        self.new_final_channels = new_final_channels
        self.g_blur = g_blur
        self.d_blur = d_blur

    def setup(self):

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
            residual_block=self.residual_block,
            blur=self.g_blur
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            sequence_count=self.sequence_count,
            final_channels=self.final_channels,
            image_size=self.image_size,
            blur=self.d_blur
        )

        self.new_netD = None
        if self.new_final_channels > 0:
            self.new_netD = Discriminator(
                input_nc=2 * self.input_nc,
                embedding_num=self.embedding_num,
                ndf=self.ndf,
                sequence_count=self.sequence_count,
                final_channels=self.new_final_channels,
                image_size=self.image_size,
                blur=self.d_blur
            )
            init_net(self.new_netD, gpu_ids=self.gpu_ids)

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        if self.new_final_channels > 0:
            self.optimizer_D = torch.optim.Adam(self.new_netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)

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
                    checkpoint = torch.load(load_path, weights_only=True)
                    net.load_state_dict(checkpoint)
                    if name=="D" and self.new_final_channels > 0:
                        for key in ["model.8.weight", "model.8.bias", "model.8.running_mean", "model.8.running_var",
                                    "model.9.weight", "model.9.bias", "model.9.running_mean", "model.9.running_var",
                                    "binary.weight", "binary.bias", "catagory.weight", "catagory.bias"]:
                            if key in checkpoint:
                                del checkpoint[key]
                        self.new_netD.load_state_dict(checkpoint, strict=False)
                        self.netD = self.new_netD
                        print("✅ 模型遷移到 final_channels=%d，開始訓練" % (self.new_final_channels))
                    else:
                        net.load_state_dict(checkpoint)
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