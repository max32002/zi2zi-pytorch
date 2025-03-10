import functools
import math
import os
import subprocess
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.init_net import init_net


# Residual Skip Connection
class ResSkip(nn.Module):
    def __init__(self, channels):
        super(ResSkip, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return x + self.relu(self.conv(x))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  
        proj_key = self.key(x).view(B, -1, H * W)  
        energy = torch.bmm(proj_query, proj_key)  
        attention = F.softmax(energy, dim=-1)  
        proj_value = self.value(x).view(B, -1, H * W)  
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)  
        return self.gamma * out + x  

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, 
                 norm_layer=nn.BatchNorm2d, layer=0, embedding_dim=128, 
                 use_dropout=False, self_attention=False, blur=False, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()
        
        self.outermost = outermost
        self.innermost = innermost
        use_bias = norm_layer != nn.BatchNorm2d  # 若使用 BatchNorm，則 bias=False

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=False)  # 這裡必須是 False
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            if use_dropout:
                self.up.add_module("dropout", nn.Dropout(0.5))

        self.submodule = submodule
        self.self_attn = SelfAttention(inner_nc) if self_attention and layer in [4, 6] else None
        self.res_skip = ResSkip(outer_nc) if not outermost and not innermost else None

    def forward(self, x, style=None):
        encoded = self.down(x)

        if self.self_attn:
            encoded = self.self_attn(encoded)

        if self.innermost:
            if style is not None:
                encoded = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), encoded], dim=1)
            decoded = self.up(encoded)
            if decoded.shape[2:] != x.shape[2:]:
                decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
            if self.res_skip:
                decoded = self.res_skip(decoded)
            return torch.cat([x, decoded], 1), encoded.view(x.shape[0], -1)

        elif self.outermost:
            if self.submodule:
                sub_output, encoded_real_A = self.submodule(encoded, style)
            else:
                sub_output = encoded
            decoded = self.up(sub_output)
            if self.res_skip:
                decoded = self.res_skip(decoded)

            return decoded, encoded_real_A

        else:
            if self.submodule:
                sub_output, encoded_real_A = self.submodule(encoded, style)
            else:
                sub_output = encoded
            decoded = self.up(sub_output)
            if decoded.shape[2:] != x.shape[2:]:
                decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
            if self.res_skip:
                decoded = self.res_skip(decoded)
            
            return torch.cat([x, decoded], 1), encoded_real_A


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, self_attention=False, blur=False):
        super(UNetGenerator, self).__init__()
        
        # 最底層（innermost），負責風格嵌入處理
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, layer=1, embedding_dim=embedding_dim, 
                                             self_attention=self_attention, blur=blur, innermost=True)

        # 中間層
        for index in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, 
                                                 norm_layer=norm_layer, layer=index+2, use_dropout=use_dropout, 
                                                 self_attention=self_attention, blur=blur)

        # 上層（恢復影像解析度）
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, layer=5, self_attention=self_attention, blur=blur)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, layer=6, self_attention=self_attention, blur=blur)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, 
                                             norm_layer=norm_layer, layer=7, self_attention=self_attention, blur=blur)

        # 最外層（outermost）
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, 
                                             norm_layer=norm_layer, layer=8, self_attention=self_attention, blur=blur, outermost=True)

        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def forward(self, x, style_or_label=None):
        """ 生成 fake_B，並獲取 encoded_real_A """
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            style = self.embedder(style_or_label)
        else:
            style = style_or_label
        
        fake_B, encoded_real_A = self.model(x, style)
        
        return fake_B, encoded_real_A

    def encode(self, x, style_or_label=None):
        """ 單純回傳編碼特徵 encoded_real_A """
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            style = self.embedder(style_or_label)
        else:
            style = style_or_label
        
        # Encoder 僅回傳 `encoded_real_A`
        _, encoded_real_A = self.model(x, style)
        return encoded_real_A

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

class Zi2ZiModel:
    def __init__(self, input_nc=1, embedding_num=40, embedding_dim=128, ngf=64, ndf=64,
                 Lconst_penalty=10, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, self_attention=False, residual_block=False, 
                 weight_decay = 1e-5, sequence_count=9, final_channels=1, beta1=0.5, g_blur=False, d_blur=False, epoch=40):

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
        self.weight_decay = weight_decay
        self.is_training = is_training
        self.image_size = image_size
        self.self_attention=self_attention
        self.residual_block=residual_block
        self.sequence_count = sequence_count
        self.final_channels = final_channels
        self.epoch = epoch
        self.g_blur = g_blur
        self.d_blur = d_blur

        device = torch.device("cuda" if self.gpu_ids and torch.cuda.is_available() else "cpu")
        self.device = device


    def setup(self):
        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
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

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        
        eta_min = 1e-6
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.epoch, eta_min=eta_min)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.epoch, eta_min=eta_min)

        self.category_loss = CategoryLoss(self.embedding_num)
        self.l1_loss = nn.L1Loss()
        self.mse = nn.MSELoss()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.l1_loss.cuda()
            self.mse.cuda()

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
        self.fake_B, self.encoded_real_A = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG.encode(self.fake_B, self.labels)
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        interpolates_logits, _ = self.netD(interpolates)
        grad_outputs = torch.ones(interpolates_logits.size(), device=self.device)
        gradients = torch.autograd.grad(
            outputs=interpolates_logits, 
            inputs=interpolates, 
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def backward_D(self, no_target_source=False):
        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B.detach()], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        real_category_loss = self.category_loss(real_category_logits, self.labels)
        fake_category_loss = self.category_loss(fake_category_logits, self.labels)
        category_loss = (real_category_loss + fake_category_loss) * self.Lcategory_penalty

        d_loss = torch.mean(F.logsigmoid(real_D_logits - fake_D_logits) + 
                             F.logsigmoid(fake_D_logits - real_D_logits))

        gp = self.compute_gradient_penalty(real_AB, fake_AB)

        gradient_penalty_weight = 10.0  # 梯度懲罰的權重
        self.d_loss = - d_loss + category_loss / 2.0 + gradient_penalty_weight * gp
        self.d_loss.backward()
        return category_loss

    def backward_G(self, no_target_source=False):
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        real_AB = torch.cat([self.real_A, self.real_B], 1)

        fake_D_logits, fake_category_logits = self.netD(fake_AB)
        real_D_logits, _ = self.netD(real_AB)

        const_loss = self.Lconst_penalty * self.mse(self.encoded_real_A, self.encoded_fake_B)
        l1_loss = self.L1_penalty * self.l1_loss(self.fake_B, self.real_B)
        fake_category_loss = self.Lcategory_penalty * self.category_loss(fake_category_logits, self.labels)
        g_loss_adv = -torch.mean(F.logsigmoid(fake_D_logits - real_D_logits))
        self.g_loss = g_loss_adv + l1_loss + fake_category_loss + const_loss
        self.g_loss.backward()
        return const_loss, l1_loss, g_loss_adv

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        new_lr_G = self.optimizer_G.param_groups[0]['lr']
        new_lr_D = self.optimizer_D.param_groups[0]['lr']
        print(f"Scheduler step executed, current step: {self.scheduler_G.last_epoch}")
        print(f"Updated learning rate: G = {new_lr_G:.6f}, D = {new_lr_D:.6f}")

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        const_loss, l1_loss, cheat_loss = self.backward_G()
        self.optimizer_G.step()
        self.forward()
        self.optimizer_G.zero_grad()
        const_loss, l1_loss, cheat_loss = self.backward_G()
        self.optimizer_G.step()
        return const_loss, l1_loss, cheat_loss

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def print_networks(self, verbose=False):
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
        assert isinstance(self.netG.state_dict(), dict), "netG.state_dict() should be a dictionary"
        assert isinstance(self.netD.state_dict(), dict), "netD.state_dict() should be a dictionary"

        torch.save(self.netG.state_dict(), os.path.join(self.save_dir, f"{epoch}_net_G.pth"))
        torch.save(self.netD.state_dict(), os.path.join(self.save_dir, f"{epoch}_net_D.pth"))
        print(f"💾 Checkpoint saved at epoch {epoch}")

    def load_networks(self, epoch):
        loaded = False
        target_filepath_G = os.path.join(self.save_dir, f"{epoch}_net_G.pth")
        target_filepath_D = os.path.join(self.save_dir, f"{epoch}_net_D.pth")
        if os.path.exists(target_filepath_G):
            try:
                self.netG.load_state_dict(torch.load(target_filepath_G, map_location=self.device, weights_only=True))
                loaded = True
            except Exception as e:
                print(f"Error loading {target_filepath_G}: {e}")
        else:
            print('file not exist:', target_filepath_G)

        if os.path.exists(target_filepath_D):
            try:
                self.netD.load_state_dict(torch.load(target_filepath_D, map_location=self.device, weights_only=True))
            except Exception as e:
                print(f"Error loading {target_filepath_D}: {e}")

        if loaded:
            print('load model %d' % epoch)
        return loaded

    def save_image(self, tensor: Union[torch.Tensor, List[torch.Tensor]]) -> None:
        grid = vutils.make_grid(tensor)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
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