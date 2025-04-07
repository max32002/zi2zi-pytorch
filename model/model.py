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

def get_unicode_codepoint(char):
    if len(char) == 1:
        return ord(char)
    elif len(char) == 2:
        high_surrogate = ord(char[0])
        low_surrogate = ord(char[1])
        return 0x10000 + (high_surrogate - 0xD800) * 0x400 + (low_surrogate - 0xDC00)
    else:
        raise ValueError("Input must be a single Unicode character or a surrogate pair.")

class StyleModulation(nn.Module):
    def __init__(self, channels, style_dim):
        super(StyleModulation, self).__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_fc = nn.Sequential(
            nn.Linear(style_dim, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x, style):
        x = self.norm(x)
        style_params = self.style_fc(style)  # (B, 2C)
        gamma, beta = style_params.chunk(2, dim=1)  # 各為 (B, C)
        gamma = gamma.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
        beta = beta.unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

class ResSkip(nn.Module):
    def __init__(self, channels):
        super(ResSkip, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)  # **改用 GroupNorm 來減少 BatchNorm 依賴**
        self.relu = nn.SiLU(inplace=True)  # **用 SiLU 替換 ReLU，減少死神經元問題**

    def forward(self, x):
        return x + self.relu(self.norm(self.conv(x)))

class LinearAttention(nn.Module):
    def __init__(self, channels, key_channels=None):
        super(LinearAttention, self).__init__()
        self.key_channels = key_channels if key_channels is not None else channels // 8
        self.value_channels = channels
        self.query = nn.Conv2d(channels, self.key_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, self.key_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, self.value_channels, kernel_size=1)
        self.out_proj = nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_mapped = self.feature_map(q).view(B, self.key_channels, N)
        k_mapped = self.feature_map(k).view(B, self.key_channels, N)
        v_reshaped = v.view(B, self.value_channels, N)
        kv_context = torch.bmm(k_mapped, v_reshaped.transpose(-1, -2))
        z_norm_factor = k_mapped.sum(dim=-1, keepdim=True)
        qkv_aggregated = torch.bmm(q_mapped.transpose(-1,-2), kv_context)
        qz_normalization = torch.bmm(q_mapped.transpose(-1,-2), z_norm_factor)
        normalized_out = (qkv_aggregated / (qz_normalization.clamp(min=1e-6))).transpose(-1,-2)
        out = normalized_out.view(B, self.value_channels, H, W)
        out = self.out_proj(out)
        return self.gamma * out + x

class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 norm_layer=nn.InstanceNorm2d, layer=0, embedding_dim=128,
                 use_dropout=False, self_attention=False, blur=False, outermost=False, innermost=False, down_stride=2):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        self.innermost = innermost
        self.embedding_dim = embedding_dim
        self.layer = layer
        self.inner_nc = inner_nc
        self.outer_nc = outer_nc
        use_bias = norm_layer != nn.BatchNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=down_stride, padding=1, bias=use_bias) # 使用 down_stride
        nn.init.kaiming_normal_(downconv.weight, nonlinearity='leaky_relu')
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)

        uprelu = nn.ReLU(inplace=False)
        upnorm = norm_layer(outer_nc)

        self.style_mod_down = None
        self.style_mod_up = None
        if not outermost:
            self.style_mod_down = StyleModulation(inner_nc, embedding_dim)
        if not innermost and not outermost:
            self.style_mod_up = StyleModulation(outer_nc, embedding_dim)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downconv)
            self.up = nn.Sequential(uprelu, upconv, nn.Tanh())
            self.attn_block = None
            self.res_skip = None
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downrelu, downconv)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.res_skip = None
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            nn.init.kaiming_normal_(upconv.weight)
            self.down = nn.Sequential(downrelu, downconv, downnorm)
            self.up = nn.Sequential(uprelu, upconv, upnorm)
            self.res_skip = ResSkip(outer_nc)
            if use_dropout:
                self.up.add_module("dropout", nn.Dropout(0.3))

        self.submodule = submodule
        #self.attn_block = SelfAttention(inner_nc) if self_attention and layer in [4, 6] else None
        self.attn_block = LinearAttention(inner_nc) if self_attention and layer in [4, 6] else None

    def _process_submodule(self, encoded, style):
        if self.submodule:
            return self.submodule(encoded, style)
        else:
            bottleneck_features = encoded.view(encoded.shape[0], -1)
            return encoded, bottleneck_features

    def _interpolate_if_needed(self, decoded, x):
        return F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False) if decoded.shape[2:] != x.shape[2:] else decoded

    def forward(self, x, style=None):
        #print(f"Input shape: {x.shape}")
        encoded_pre_attn = self.down(x)
        #print(f"encoded_pre_attn shape: {encoded_pre_attn.shape if encoded_pre_attn is not None else None}")

        encoded_modulated_down = encoded_pre_attn
        if self.style_mod_down and style is not None and not self.outermost:
            # print(f"Shape of encoded_pre_attn before StyleModulation (down): {encoded_pre_attn.shape}, style shape: {style.shape}")
            encoded_modulated_down = self.style_mod_down(encoded_pre_attn, style)
            # print(f"encoded after style_mod_down shape: {encoded_modulated_down.shape}")

        encoded_post_attn = encoded_modulated_down
        if self.attn_block:
            encoded_post_attn = self.attn_block(encoded_pre_attn)
            #print(f"encoded_post_attn shape: {encoded_post_attn.shape}")

        # print(f"encoded_post_attn shape: {encoded_post_attn.shape if encoded_post_attn is not None else None}")
        if self.innermost:
            bottleneck_features_flat = encoded_post_attn.view(x.shape[0], -1)
            decoded = self.up(encoded_post_attn)
            decoded = self._interpolate_if_needed(decoded, x)
            return torch.cat([x, decoded], 1), bottleneck_features_flat
        else: # Intermediate or Outermost blocks
            sub_output_cat, bottleneck_features_flat = self._process_submodule(encoded_post_attn, style)
            decoded = self.up(sub_output_cat)
            decoded = self._interpolate_if_needed(decoded, x)

            decoded_modulated_up = decoded
            if self.style_mod_up is not None and style is not None and not self.outermost:
                decoded_modulated_up = self.style_mod_up(decoded, style)

            if self.res_skip and not self.outermost:
                decoded_modulated_up = self.res_skip(decoded_modulated_up)

            if self.outermost:
                return decoded_modulated_up, bottleneck_features_flat
            else:
                return torch.cat([x, decoded_modulated_up], 1), bottleneck_features_flat
                
class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, num_downs=8, ngf=64, embedding_num=40, embedding_dim=128,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False, self_attention=False, blur=False):
        super(UNetGenerator, self).__init__()

        # 最底層（innermost）
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                             norm_layer=norm_layer, layer=1, embedding_dim=embedding_dim,
                                             self_attention=self_attention, blur=blur, innermost=True, down_stride=1) # 修改 down_stride

        # 中間層
        for index in range(num_downs - 5):
            stride = 2 if index < num_downs - 6 else 1 # 只在較前的層下採樣
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, layer=index+2, use_dropout=use_dropout,
                                                 self_attention=self_attention, blur=blur, embedding_dim=embedding_dim, down_stride=stride) # 添加 down_stride

        # 上層（恢復影像解析度）
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, layer=5, self_attention=self_attention, blur=blur, embedding_dim=embedding_dim)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, layer=6, self_attention=self_attention, blur=blur, embedding_dim=embedding_dim)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, layer=7, self_attention=self_attention, blur=blur, embedding_dim=embedding_dim)

        # 最外層（outermost）
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                             norm_layer=norm_layer, layer=8, self_attention=self_attention, blur=blur, outermost=True, embedding_dim=embedding_dim)

        self.embedder = nn.Embedding(embedding_num, embedding_dim)

    def _prepare_style(self, style_or_label):
        if style_or_label is not None and isinstance(style_or_label, torch.Tensor):
             if 'LongTensor' in style_or_label.type():
                  if style_or_label.max() >= self.embedder.num_embeddings:
                       raise ValueError(f"Label index {style_or_label.max()} is out of bounds for embedding_num={self.embedder.num_embeddings}")
                  return self.embedder(style_or_label)
             else:
                  return style_or_label
        elif style_or_label is None:
             raise ValueError("Style/label cannot be None for StyleModulation U-Net.")
        else: # Handle non-tensor case? Maybe raise error.
             raise TypeError(f"Unsupported type for style_or_label: {type(style_or_label)}")

    def forward(self, x, style_or_label=None):
        style = self._prepare_style(style_or_label)
        fake_B, encoded_bottleneck = self.model(x, style)
        return fake_B, encoded_bottleneck

    def encode(self, x, style_or_label=None):
        style = self._prepare_style(style_or_label)
        _, encoded_bottleneck = self.model(x, style)
        return encoded_bottleneck

class Discriminator(nn.Module):
    def __init__(self, input_nc, embedding_num, ndf=64, norm_layer=nn.BatchNorm2d, blur=False):
        super(Discriminator, self).__init__()

        use_bias = norm_layer != nn.BatchNorm2d
        kw = 5
        padw = 2

        sequence = [
            nn.utils.spectral_norm(nn.Conv2d(2, ndf, kernel_size=kw, stride=2, padding=padw)), # 將 input_nc 改為 2
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
    def __init__(self, input_nc=1, embedding_num=2, embedding_dim=128, ngf=64, ndf=64,
                 Lconst_penalty=10, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, gpu_ids=None, save_dir='.', is_training=True,
                 self_attention=False, residual_block=False, 
                 weight_decay = 1e-5, beta1=0.5, g_blur=False, d_blur=False, epoch=40,
                 gradient_clip=0.5, norm_type="instance"):

        self.norm_type = norm_type

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        # Loss weights
        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty
        self.perceptual_weight = 10.0 # Make perceptual weight configurable?
        self.gradient_penalty_weight = 10.0 # Make GP weight configurable?

        self.epoch = epoch
        self.schedule = schedule

        self.save_dir = save_dir
        self.gpu_ids = gpu_ids
        self.device = torch.device("cuda" if self.gpu_ids and torch.cuda.is_available() else "cpu")

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
        #self.residual_block=residual_block
        self.g_blur = g_blur
        self.d_blur = d_blur
        self.gradient_clip = gradient_clip

        self.setup()

        self.scaler_G = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())
        self.scaler_D = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    def setup(self):
        if self.norm_type.lower() == "batch":
            norm_layer_g = nn.BatchNorm2d
        elif self.norm_type.lower() == "instance":
            norm_layer_g = nn.InstanceNorm2d
        else:
            raise ValueError(f"Unsupported norm_type for Generator: {self.norm_type}")
        norm_layer_d = nn.BatchNorm2d

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc, # Output is single channel like input
            ngf=self.ngf,
            use_dropout=self.use_dropout,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            self_attention=self.self_attention,
            blur=self.g_blur,
            norm_layer=norm_layer_g # Pass chosen norm layer
        ).to(self.device)

        # --- Initialize Discriminator ---
        self.netD = Discriminator(
            input_nc=self.input_nc, # D sees concatenated input (A+B = 1+1=2 channels)
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            blur=self.d_blur,
            norm_layer=norm_layer_d # Use specified norm for D
        ).to(self.device)

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.AdamW(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        self.optimizer_D = torch.optim.AdamW(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999), weight_decay=self.weight_decay)
        print(f"Optimizers: AdamW (lr={self.lr}, beta1={self.beta1}, wd={self.weight_decay})")

        eta_min = self.lr * 0.01 # Example: anneal down to 1% of initial LR
        self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_G, T_max=self.epoch, eta_min=eta_min)
        self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_D, T_max=self.epoch, eta_min=eta_min)
        print(f"Schedulers: CosineAnnealingLR (T_max={self.epoch}, eta_min={eta_min})")

        self.criterion_L1 = nn.L1Loss().to(self.device)
        self.criterion_MSE = nn.MSELoss().to(self.device) # For const_loss
        self.criterion_Category = CategoryLoss(self.embedding_num).to(self.device)
        self.criterion_Perceptual = PerceptualLoss().to(self.device) # Assumes VGG normalization inside
        self.criterion_FeatureMatch = nn.L1Loss().to(self.device) # For Feature Matching Loss

        print("Loss functions initialized.")

        # Set training/eval mode
        self.set_train_eval_mode()

    def set_train_eval_mode(self):
        if self.is_training:
            self.netG.train()
            self.netD.train()
            print("Model set to TRAIN mode.")
        else:
            self.netG.eval()
            self.netD.eval()
            print("Model set to EVAL mode.")

    def set_input(self, data):
        self.labels = data['label'].to(self.device)
        self.real_A = data['A'].to(self.device) # Input font image
        self.real_B = data['B'].to(self.device) # Target font image

    def forward(self):
        self.fake_B, self.encoded_real_A = self.netG(self.real_A, self.labels)
        self.encoded_fake_B = self.netG.encode(self.fake_B, self.labels)

    def compute_feature_matching_loss(self, real_AB, fake_AB):
        fm_loss = 0.0
        real_fm = self.netD.model(torch.cat([self.real_A, self.real_B], 1))
        fake_fm = self.netD.model(torch.cat([self.real_A, self.fake_B], 1))
        fm_loss = self.criterion_FeatureMatch(fake_fm, real_fm.detach()) # Detach real features
        return fm_loss * 10.0 # Add weight to FM loss (common practice)

    def compute_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates, _ = self.netD(interpolates) # Get discriminator output for interpolates
        grad_outputs = torch.ones(d_interpolates.size(), device=self.device, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1) # Flatten gradients
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def backward_D(self, no_target_source=False):
        real_AB = torch.cat([self.real_A, self.real_B], 1)
        fake_AB = torch.cat([self.real_A, self.fake_B.detach()], 1)

        real_D_logits, real_category_logits = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        loss_D_adv = -torch.mean(F.logsigmoid(real_D_logits - fake_D_logits) +
                                  F.logsigmoid(fake_D_logits - real_D_logits))

        loss_D_real_category = self.criterion_Category(real_category_logits, self.labels)
        loss_D_fake_category = self.criterion_Category(fake_category_logits, self.labels)
        loss_D_category = (loss_D_real_category + loss_D_fake_category) * 0.5 # Average category loss

        gp = self.compute_gradient_penalty(real_AB, fake_AB) # Keep it for now

        self.loss_D = loss_D_adv + \
                      loss_D_category * self.Lcategory_penalty + \
                      gp * self.gradient_penalty_weight

        return loss_D_adv, loss_D_category, gp

    def backward_G(self, no_target_source=False):
        real_AB = torch.cat([self.real_A, self.real_B], 1).detach() # Detach real pair
        fake_AB = torch.cat([self.real_A, self.fake_B], 1)
        real_D_logits, _ = self.netD(real_AB)
        fake_D_logits, fake_category_logits = self.netD(fake_AB)

        loss_G_adv = -torch.mean(F.logsigmoid(fake_D_logits - real_D_logits) +
                                  F.logsigmoid(real_D_logits - fake_D_logits))

        loss_G_category = self.criterion_Category(fake_category_logits, self.labels)

        loss_G_L1 = self.criterion_L1(self.fake_B, self.real_B)

        loss_G_const = self.criterion_MSE(self.encoded_fake_B, self.encoded_real_A.detach()) # Detach real encoding

        loss_G_FM = self.compute_feature_matching_loss(real_AB, fake_AB)

        loss_G_perceptual = self.criterion_Perceptual(self.fake_B, self.real_B)

        self.loss_G = loss_G_adv + \
                      loss_G_category * self.Lcategory_penalty + \
                      loss_G_L1 * self.L1_penalty + \
                      loss_G_const * self.Lconst_penalty + \
                      loss_G_FM + \
                      loss_G_perceptual * self.perceptual_weight

        return loss_G_adv, loss_G_category, loss_G_L1, loss_G_const, loss_G_FM, loss_G_perceptual

    def optimize_parameters(self, use_autocast=True):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()

        loss_D_adv, loss_D_category, gp = 0, 0, 0
        if use_autocast:
            with torch.amp.autocast(device_type='cuda'):
                loss_D_adv, loss_D_category, gp = self.backward_D()
                scaled_loss_D = self.scaler_D.scale(self.loss_D)
                scaled_loss_D.backward()
                self.scaler_D.unscale_(self.optimizer_D)
                self.scaler_D.step(self.optimizer_D)
                self.scaler_D.update()
        else:
            loss_D_adv, loss_D_category, gp = self.backward_D()
            self.loss_D.backward()
            grad_norm_d = torch.nn.utils.clip_grad_norm_(self.netD.parameters(), self.gradient_clip)
            self.optimizer_D.step()

        if torch.isnan(self.loss_D):
             print("ERROR: Discriminator loss is NaN. Stopping training.")
             raise RuntimeError("Discriminator loss is NaN")

        self.set_requires_grad(self.netD, False) # Disable grads for D
        self.optimizer_G.zero_grad()
        self.forward() # Forward pass needed again for G loss components

        if use_autocast:
            with torch.amp.autocast(device_type='cuda'):
                loss_G_adv, loss_G_category, loss_G_L1, loss_G_const, loss_G_FM, loss_G_perceptual = self.backward_G()
                scaled_loss_G = self.scaler_G.scale(self.loss_G)
                scaled_loss_G.backward()
                self.scaler_G.unscale_(self.optimizer_G)
                torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
                self.scaler_G.step(self.optimizer_G)
                self.scaler_G.update()
        else:
            loss_G_adv, loss_G_category, loss_G_L1, loss_G_const, loss_G_FM, loss_G_perceptual = self.backward_G()
            self.loss_G.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.netG.parameters(), self.gradient_clip)
            self.optimizer_G.step()


        if torch.isnan(self.loss_G):
             print("ERROR: Generator loss is NaN. Stopping training.")
             raise RuntimeError("Generator loss is NaN")

        return {
            'G_adv': loss_G_adv.item(),
            'G_category': loss_G_category.item(),
            'G_L1': loss_G_L1.item(),
            'G_const': loss_G_const.item(),
            'G_FM': loss_G_FM.item(),
            'G_perceptual': loss_G_perceptual.item(),
            'G_total': self.loss_G.item(),
            'D_adv': loss_D_adv.item(),
            'D_category': loss_D_category.item(),
            'D_gp': gp.item(),
            'D_total': self.loss_D.item()
        }

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

    def process_image(self, image, crop_src_font, canvas_size, resize_canvas, anti_aliasing_strength, binary_image):
        """處理圖像：裁剪、縮放、抗鋸齒、二值化"""
        if crop_src_font:
            image = image[0:canvas_size, 0:canvas_size]
            if resize_canvas > 0 and canvas_size != resize_canvas:
                image = cv2.resize(image, (resize_canvas, resize_canvas), interpolation=cv2.INTER_LANCZOS4)
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

    def sample(self, batch, basename, src_char_list=None, crop_src_font=False, canvas_size=256, resize_canvas=256,
               filename_rule="seq", binary_image=True, anti_aliasing_strength=1, image_ext="png"):
        """生成並儲存圖像樣本"""
        with torch.no_grad():
            self.set_input(batch[0], batch[2], batch[1])
            self.forward()

            output_images = torch.cat([self.fake_B, self.real_B], 3)
            for i, (label, image_tensor) in enumerate(zip(batch[0], output_images)):
                label_dir = os.path.join(basename, str(label.item()))
                os.makedirs(label_dir, exist_ok=True)  # 確保目錄存在

                if filename_rule == "seq":
                    filename = str(i)
                elif src_char_list and i < len(src_char_list):
                    if filename_rule == "char":
                        filename = src_char_list[i]
                    elif filename_rule == "unicode_hex":
                        filename = hex(get_unicode_codepoint(src_char_list[i]))[2:]
                    elif filename_rule == "unicode_int":
                        filename = str(get_unicode_codepoint(src_char_list[i]))
                else:
                    filename = str(i)  # 如果 src_char_list 不存在或長度不夠，使用序列號

                opencv_image = cv2.cvtColor(self.save_image(image_tensor), cv2.COLOR_BGR2GRAY)
                processed_image = self.process_image(opencv_image, crop_src_font, canvas_size, resize_canvas,
                                                    anti_aliasing_strength, binary_image)
                self.save_image_to_disk(processed_image, label_dir, filename, image_ext)

