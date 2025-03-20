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
import torchvision.models as models
import torchvision.transforms as T
import torchvision.utils as vutils
from PIL import Image
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import VGG16_Weights

from utils.init_net import init_net

def get_unicode_codepoint(char):
    if len(char) == 1:
        return ord(char)
    elif len(char) == 2:
        high_surrogate = ord(char[0])
        low_surrogate = ord(char[1])
        return 0x10000 + (high_surrogate - 0xD800) * 0x400 + (low_surrogate - 0xDC00)
    else:
        raise ValueError("Input must be a single Unicode character or a surrogate pair.")

class ResSkip(nn.Module):
    def __init__(self, channels):
        super(ResSkip, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)  # **改用 GroupNorm 來減少 BatchNorm 依賴**
        self.relu = nn.SiLU(inplace=True)  # **用 SiLU 替換 ReLU，減少死神經元問題**

    def forward(self, x):
        return x + self.relu(self.norm(self.conv(x)))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)
        self.scale = (channels // 8) ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, H * W)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy * self.scale, dim=-1)
        proj_value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 norm_layer=nn.InstanceNorm2d, layer=0, embedding_dim=128,
                 use_dropout=False, self_attention=False, blur=False, outermost=False, innermost=False):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost
        use_bias = norm_layer != nn.BatchNorm2d

        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        nn.init.kaiming_normal_(downconv.weight, nonlinearity='leaky_relu')
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=False)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + embedding_dim, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            if use_dropout:
                self.up.add_module("dropout", nn.Dropout(0.3))

        self.submodule = submodule
        self.self_attn = SelfAttention(inner_nc) if self_attention and layer in [4, 6] else None
        self.res_skip = ResSkip(outer_nc) if not outermost and not innermost else None

    def _process_submodule(self, encoded, style):
        if self.submodule:
            return self.submodule(encoded, style)
        else:
            return encoded, None

    def _interpolate_if_needed(self, decoded, x):
        return F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False) if decoded.shape[2:] != x.shape[2:] else decoded

    def forward(self, x, style=None):
        encoded = self.down(x)

        if self.self_attn:
            encoded = self.self_attn(encoded)

        if self.innermost:
            if style is not None:
                encoded = torch.cat([style.view(style.shape[0], style.shape[1], 1, 1), encoded], dim=1)
            decoded = self.up(encoded)
            decoded = self._interpolate_if_needed(decoded, x)
            if self.res_skip:
                decoded = self.res_skip(decoded)
            return torch.cat([x, decoded], 1), encoded.view(x.shape[0], -1)

        sub_output, encoded_real_A = self._process_submodule(encoded, style)

        decoded = self.up(sub_output)
        decoded = self._interpolate_if_needed(decoded, x)

        if self.res_skip:
            decoded = self.res_skip(decoded)

        if self.outermost:
            return decoded, encoded_real_A
        else:
            return torch.cat([x, decoded], 1), encoded_real_A

class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, self_attention=False, blur=False):
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

    def _prepare_style(self, style_or_label):
        if style_or_label is not None and 'LongTensor' in style_or_label.type():
            return self.embedder(style_or_label)
        else:
            return style_or_label

    def forward(self, x, style_or_label=None):
        style = self._prepare_style(style_or_label)
        fake_B, encoded_real_A = self.model(x, style)
        return fake_B, encoded_real_A

    def encode(self, x, style_or_label=None):
        style = self._prepare_style(style_or_label)
        _, encoded_real_A = self.model(x, style)
        return encoded_real_A

class Discriminator(nn.Module):
    def __init__(self, input_nc, embedding_num, ndf=64, norm_layer=nn.BatchNorm2d, blur=False):
        super(Discriminator, self).__init__()
        
        use_bias = norm_layer != nn.BatchNorm2d
        kw = 5
        padw = 2

        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, 3):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        sequence += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        self.model = nn.Sequential(*sequence)
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # 自適應池化
        final_features = ndf * nf_mult * 4 * 4
        
        self.binary = nn.Linear(final_features, 1)
        self.category = nn.Linear(final_features, embedding_num)
        
        self.blur = blur
        if blur:
            self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=1.0)

    def forward(self, input):
        if self.blur:
            input = self.gaussian_blur(input)
        
        features = self.model(input)
        features = self.global_pool(features)
        features = features.view(input.shape[0], -1)  # 展平成 batch x final_features
        
        binary_logits = self.binary(features)
        category_logits = self.category(features)
        return binary_logits, category_logits

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

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg[:4]))   # Conv1_2 (Input: 3 channels -> 64 channels)
        self.slice2 = nn.Sequential(*list(vgg[4:9]))  # Conv2_2 (Input: 64 channels -> 128 channels)
        self.slice3 = nn.Sequential(*list(vgg[9:16])) # Conv3_3 (Input: 128 channels -> 256 channels)
        self.slice4 = nn.Sequential(*list(vgg[16:23]))# Conv4_3 (Input: 256 channels -> 512 channels)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """將灰階 (1 通道) 轉換為 RGB (3 通道)，再傳入 VGG 模型"""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (N, 1, H, W) -> (N, 3, H, W)
            y = y.repeat(1, 3, 1, 1)

        # 確保輸入尺寸符合 VGG 預期
        x1, y1 = self.slice1(x), self.slice1(y)  # (N, 64, H, W)
        x2, y2 = self.slice2(x1), self.slice2(y1)  # (N, 128, H, W)
        x3, y3 = self.slice3(x2), self.slice3(y2)  # (N, 256, H, W)
        x4, y4 = self.slice4(x3), self.slice4(y3)  # (N, 512, H, W)

        loss = (
            nn.functional.l1_loss(x1, y1) +
            nn.functional.l1_loss(x2, y2) +
            nn.functional.l1_loss(x3, y3) +
            nn.functional.l1_loss(x4, y4)
        )
        return loss

class Zi2ZiModel:
    def __init__(self, input_nc=1, embedding_num=40, embedding_dim=128, ngf=64, ndf=64,
                 Lconst_penalty=10, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 self_attention=False, residual_block=False, 
                 weight_decay = 1e-5, beta1=0.5, g_blur=False, d_blur=False, epoch=40,
                 gradient_clip=0.5, norm_type="instance"):

        self.norm_type = norm_type  # 保存 norm_type

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty

        self.epoch = epoch
        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids
        device = torch.device("cuda" if self.gpu_ids and torch.cuda.is_available() else "cpu")
        self.device = device

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.is_training = is_training
        self.self_attention=self_attention
        self.residual_block=residual_block
        self.g_blur = g_blur
        self.d_blur = d_blur


        self.scaler_G = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.scaler_D = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.feature_matching_loss = nn.L1Loss()
        self.gradient_clip = gradient_clip

    def setup(self):
        if self.norm_type == "batch":
            norm_layer = nn.BatchNorm2d
        else:  # 預設或 instance
            norm_layer = nn.InstanceNorm2d

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
            blur=self.g_blur,
            norm_layer=norm_layer
        )
        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            blur=self.d_blur,
            norm_layer=nn.BatchNorm2d
        )

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        
        eta_min = 1e-6
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.epoch, eta_min=eta_min)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.epoch, eta_min=eta_min)

        self.vgg_loss = PerceptualLoss().to(self.device)
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

    def compute_feature_matching_loss(self, real_AB, fake_AB):
        real_features = self.netD.model[:-1](real_AB)
        fake_features = self.netD.model[:-1](fake_AB)
        return self.feature_matching_loss(real_features, fake_features)

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

        gradient_penalty_weight = 10.0
        self.d_loss = - d_loss + category_loss / 2.0 + gradient_penalty_weight * gp

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

        fm_loss = self.compute_feature_matching_loss(real_AB, fake_AB)

        self.g_loss = g_loss_adv + l1_loss + fake_category_loss + const_loss + fm_loss
        
        perceptual_loss = self.vgg_loss(self.fake_B, self.real_B)
        perceptual_weight = 10.0  # 感知損失的權重
        self.g_loss += perceptual_weight * perceptual_loss

        return const_loss, l1_loss, g_loss_adv, fm_loss, perceptual_loss

    def optimize_parameters(self, use_autocast=False):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        if use_autocast:
            with torch.amp.autocast(device_type='cuda'):
                category_loss = self.backward_D()
                scaled_d_loss = self.scaler_D.scale(self.d_loss)
                scaled_d_loss.backward()
                self.scaler_D.unscale_(self.optimizer_D)
                grad_norm_d = torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.gradient_clip)
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()
        else:
            category_loss = self.backward_D()
            self.d_loss.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.gradient_clip)
            self.optimizer_D.step()

        # 檢查判別器損失是否為 NaN
        if torch.isnan(self.d_loss):
            print("判別器損失為 NaN，停止訓練。")
            return  # 或執行其他適當的錯誤處理

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss = 0, 0, 0, 0, 0

        if use_autocast:
            with torch.amp.autocast(device_type='cuda'):
                const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss = self.backward_G()
                scaled_g_loss = self.scaler_G.scale(self.g_loss)
                scaled_g_loss.backward()
                self.scaler_G.unscale_(self.optimizer_G)
                grad_norm_g = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
        else:
            const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss = self.backward_G()
            self.g_loss.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
            self.optimizer_G.step()

        self.forward()
        self.optimizer_G.zero_grad()

        if use_autocast:
            with torch.amp.autocast(device_type='cuda'):
                const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss = self.backward_G()
                scaled_g_loss = self.scaler_G.scale(self.g_loss)
                scaled_g_loss.backward()
                self.scaler_G.unscale_(self.optimizer_G)
                grad_norm_g = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
        else:
            const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss = self.backward_G()
            self.g_loss.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
            self.optimizer_G.step()

        # 可以選擇性地監控梯度範數
        # print(f"判別器梯度範數：{grad_norm_d}")
        # print(f"生成器梯度範數：{grad_norm_g}")

        return const_loss, l1_loss, cheat_loss, fm_loss, perceptual_loss

    def update_lr(self):
        self.scheduler_G.step()
        self.scheduler_D.step()
        new_lr_G = self.optimizer_G.param_groups[0]['lr']
        new_lr_D = self.optimizer_D.param_groups[0]['lr']
        print(f"Scheduler step executed, current step: {self.scheduler_G.last_epoch}")
        print(f"Updated learning rate: G = {new_lr_G:.6f}, D = {new_lr_D:.6f}")

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
                self.netG.load_state_dict(torch.load(target_filepath_G, map_location=self.device), strict=False)
                loaded = True
            except Exception as e:
                print(f"Error loading {target_filepath_G}: {e}")
        else:
            print(f"File not found: {target_filepath_G}")

        if os.path.exists(target_filepath_D):
            try:
                state_dict = torch.load(target_filepath_D, map_location=self.device)
                self.netD.load_state_dict(state_dict, strict=False)  # 忽略形狀不匹配的層
                self._initialize_unmatched_weights(self.netD, state_dict)  # 初始化未載入的層
            except Exception as e:
                print(f"Error loading {target_filepath_D}: {e}")

        if loaded:
            print(f"✅ Model {epoch} loaded successfully")
        return loaded

    def _initialize_unmatched_weights(self, model, loaded_state_dict):
        """ 初始化 `netD` 中未載入的層 """
        for name, param in model.named_parameters():
            if name not in loaded_state_dict:
                print(f"🔄 Re-initializing layer: {name}")
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='leaky_relu')
                elif "bias" in name:
                    nn.init.constant_(param, 0)

        for name, buffer in model.named_buffers():
            if name not in loaded_state_dict:
                print(f"🔄 Re-initializing buffer: {name}")
                buffer.zero_()

    def save_image(self, tensor: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        """將張量轉換為 OpenCV 圖像"""
        grid = vutils.make_grid(tensor)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return ndarr

    def anti_aliasing(self, image, strength=1):
        """抗鋸齒處理"""
        ksize = max(1, strength * 2 + 1)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred

    def process_image(self, image, crop_src_font, canvas_size, resize_canvas_size, anti_aliasing_strength, binary_image):
        """處理圖像：裁剪、縮放、抗鋸齒、二值化"""
        if crop_src_font:
            image = image[0:canvas_size, 0:canvas_size]
            if resize_canvas_size > 0 and canvas_size != resize_canvas_size:
                image = cv2.resize(image, (resize_canvas_size, resize_canvas_size), interpolation=cv2.INTER_LANCZOS4)
            else:
                image = cv2.resize(image, (canvas_size * 2, canvas_size * 2), interpolation=cv2.INTER_CUBIC)
                image = self.anti_aliasing(image, 1)
                image = cv2.resize(image, (canvas_size, canvas_size), interpolation=cv2.INTER_CUBIC)
            image = self.anti_aliasing(image, anti_aliasing_strength)

        if binary_image:
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def save_image_to_disk(self, image, label_dir, filename, image_ext):
        """將圖像儲存到磁碟，並根據需要轉換為 SVG"""
        save_path = os.path.join(label_dir, f"{filename}.{image_ext}")
        if image_ext == "svg":
            save_path_pgm = os.path.join(label_dir, f"{filename}.pgm")
            cv2.imwrite(save_path_pgm, image)
            subprocess.call(['potrace', '-b', 'svg', '-u', '60', save_path_pgm, '-o', save_path])
            os.remove(save_path_pgm)
        else:
            cv2.imwrite(save_path, image)

    def sample(self, batch, basename, src_char_list=None, crop_src_font=False, canvas_size=256, resize_canvas_size=256,
               filename_mode="seq", binary_image=True, anti_aliasing_strength=1, image_ext="png"):
        """生成並儲存圖像樣本"""
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()

            output_images = torch.cat([self.fake_B, self.real_B], 3)
            for i, (label, image_tensor) in enumerate(zip(batch[0], output_images)):
                label_dir = os.path.join(basename, str(label.item()))
                os.makedirs(label_dir, exist_ok=True)  # 確保目錄存在

                if filename_mode == "seq":
                    filename = str(i)
                elif src_char_list and i < len(src_char_list):
                    if filename_mode == "char":
                        filename = src_char_list[i]
                    elif filename_mode == "unicode_hex":
                        filename = hex(get_unicode_codepoint(src_char_list[i]))[2:]
                    elif filename_mode == "unicode_int":
                        filename = str(get_unicode_codepoint(src_char_list[i]))
                else:
                    filename = str(i)  # 如果 src_char_list 不存在或長度不夠，使用序列號

                opencv_image = cv2.cvtColor(self.save_image(image_tensor), cv2.COLOR_BGR2GRAY)
                processed_image = self.process_image(opencv_image, crop_src_font, canvas_size, resize_canvas_size,
                                                    anti_aliasing_strength, binary_image)
                self.save_image_to_disk(processed_image, label_dir, filename, image_ext)

