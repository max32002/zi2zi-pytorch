import functools
import os
import subprocess
import sys
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from utils.init_net import init_net

from .discriminators import Discriminator
from .generators import UNetGenerator
from .losses import BinaryLoss, CategoryLoss

def get_unicode_codepoint(char):
    if sys.maxunicode >= 0x10FFFF:
        # 直接處理單一字元
        return ord(char)
    else:
        # 針對 UCS-2 需要特別處理代理對
        if len(char) == 2:
            high, low = map(ord, char)
            return (high - 0xD800) * 0x400 + (low - 0xDC00) + 0x10000
        else:
            return ord(char)

class Zi2ZiModel:
    def __init__(self, input_nc=1, embedding_num=40, embedding_dim=128,
                 ngf=64, ndf=64,
                 lambda_adv=0.25,
                 accum_steps=1,
                 Lconst_penalty=15, Lcategory_penalty=1, L1_penalty=100,
                 schedule=10, lr=0.001, lr_D=None, gpu_ids=None, save_dir='.', is_training=True,
                 image_size=256, self_attention=False, d_spectral_norm=False, norm_type="instance"):

        self.gpu_ids = gpu_ids
        self.device = torch.device("cuda" if self.gpu_ids and torch.cuda.is_available() else "cpu")

        if is_training:
            self.use_dropout = True
        else:
            self.use_dropout = False

        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.Lconst_penalty = Lconst_penalty
        self.Lcategory_penalty = Lcategory_penalty
        self.L1_penalty = L1_penalty
        self.lambda_adv = lambda_adv
        self.schedule = schedule
        self.save_dir = save_dir
        self.gpu_ids = gpu_ids

        self.input_nc = input_nc
        self.embedding_dim = embedding_dim
        self.embedding_num = embedding_num
        self.ngf = ngf
        self.ndf = ndf
        self.lr = lr
        self.lr_D = lr_D if lr_D is not None else lr
        self.is_training = is_training
        self.image_size = image_size
        self.self_attention = self_attention
        self.d_spectral_norm = d_spectral_norm
        self.norm_type = norm_type
        self.accum_steps = max(1, accum_steps)
        self._accum_counter = 0

        self.setup()

    def setup(self):
        # choose norm
        if self.norm_type == 'batch':
            norm_layer = nn.BatchNorm2d
        elif self.norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % self.norm_type)

        # build nets (assumes UNetGenerator and Discriminator are defined and imported)
        num_downs = 8 if self.image_size != 384 else 7

        self.netG = UNetGenerator(
            input_nc=self.input_nc,
            output_nc=self.input_nc,
            embedding_num=self.embedding_num,
            embedding_dim=self.embedding_dim,
            ngf=self.ngf,
            norm_layer=norm_layer,
            use_dropout=self.use_dropout,
            num_downs=num_downs,
            self_attention=self.self_attention
        ).to(self.device)

        self.netD = Discriminator(
            input_nc=2 * self.input_nc,
            embedding_num=self.embedding_num,
            ndf=self.ndf,
            norm_layer=norm_layer,
            image_size=self.image_size,
            use_spectral_norm=self.d_spectral_norm
        ).to(self.device)

        init_net(self.netG, gpu_ids=self.gpu_ids)
        init_net(self.netD, gpu_ids=self.gpu_ids)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr_D, betas=(0.5, 0.999))

        self.category_loss = CategoryLoss(self.embedding_num)
        self.real_binary_loss = BinaryLoss(True)
        self.fake_binary_loss = BinaryLoss(False)
        self.fake_binary_loss = BinaryLoss(False)
        self.l1_loss = nn.L1Loss()

        self.mse = nn.MSELoss()
        self.sigmoid = nn.Sigmoid()

        if self.gpu_ids:
            self.category_loss.cuda()
            self.real_binary_loss.cuda()
            self.fake_binary_loss.cuda()
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

    def set_input(self, data):
        self.model_input_data = data
        self.labels = data['label'].to(self.device)
        self.real_A = data['A'].to(self.device) # Input font image
        self.real_B = data['B'].to(self.device) # Target font image

    def forward(self, data):
        real_A = data['A'].to(self.device)
        real_B = data['B'].to(self.device)
        labels = data['label'].to(self.device)

        # 生成器 forward（取得 fake_B 及 deep feature）
        fake_B, fake_B_emb = self.netG(real_A, labels, return_feat=True)

        # 取得 real_B 的 embedding
        _, real_B_emb = self.netG(real_B, labels, return_feat=True)

        # 1) Reconstruction loss
        self.loss_l1 = self.l1_loss(fake_B, real_B)

        # 2) Feature Constancy loss（使用 embedding，不再第二次 forward）
        self.loss_const = self.l1_loss(fake_B_emb, real_B_emb)

        # Store for access
        self.fake_B = fake_B
        self.encoded_fake_B = fake_B_emb
        self.labels = labels
        self.real_A = real_A
        self.real_B = real_B

        return {
            "fake_B": fake_B,
            "fake_B_emb": fake_B_emb,
            "real_B_emb": real_B_emb
        }

    def update_lambda_adv(self):
        """
        Dynamically adjust lambda_adv based on discriminator loss.
        Designed for zi2zi-style font GAN where D can easily overpower G.
        """
        if not hasattr(self, "lambda_adv"):
            return

        d = float(self.d_loss.item())

        # --- heuristic schedule ---
        if d < 0.02:
            self.lambda_adv = 0.20
        elif d < 0.08:
            self.lambda_adv = 0.35
        elif d < 0.20:
            self.lambda_adv = 0.50
        else:
            self.lambda_adv = 0.60

    def optimize_parameters(self):
        # 1. Forward G
        self.forward(self.model_input_data)

        real_A = self.real_A
        real_B = self.real_B
        fake_B = self.fake_B
        labels = self.labels

        fake_AB = torch.cat([real_A, fake_B], 1)
        real_AB = torch.cat([real_A, real_B], 1)

        # 2. Update D (gradient accumulation)
        if self._accum_counter == 0:
            self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad(set_to_none=True)

        # Forward D with detached fake
        pred_fake_d, fake_category_logits_d = self.netD(fake_AB.detach())
        pred_real, real_category_logits = self.netD(real_AB)

        loss_D_real = self.real_binary_loss(pred_real)
        loss_D_fake = self.fake_binary_loss(pred_fake_d)

        real_category_loss = self.category_loss(real_category_logits, labels)
        fake_category_loss_d = self.category_loss(fake_category_logits_d, labels)
        self.category_loss_D = (real_category_loss + fake_category_loss_d) * self.Lcategory_penalty

        self.d_loss = ((loss_D_real + loss_D_fake) * 0.5 + self.category_loss_D * 0.5) / self.accum_steps
        self.d_loss.backward()

        self.update_lambda_adv()

        # 3. Update G
        self.set_requires_grad(self.netD, False)
        if self._accum_counter == 0:
            self.optimizer_G.zero_grad(set_to_none=True)

        # Forward D again with attached fake (using updated weights is standard, or use retained graph?)
        # Standard GAN training uses updated D for G loss.
        pred_fake, fake_category_logits = self.netD(fake_AB)

        self.loss_G_GAN = self.real_binary_loss(pred_fake)
        fake_category_loss_G = self.category_loss(fake_category_logits, labels) * self.Lcategory_penalty

        self.g_loss = (
            self.loss_G_GAN * self.lambda_adv +
            self.loss_l1 * self.L1_penalty +
            self.loss_const * self.Lconst_penalty +
            fake_category_loss_G
        )

        self.g_loss = self.g_loss / self.accum_steps
        self.g_loss.backward()

        self._accum_counter += 1
        if self._accum_counter >= self.accum_steps:
            print("match lost update.")
            self.optimizer_D.step()
            self.optimizer_G.step()
            self.optimizer_D.zero_grad(set_to_none=True)
            self.optimizer_G.zero_grad(set_to_none=True)
            self._accum_counter = 0

        # Return losses for logging
        return {
            "loss_const": self.loss_const.item(),
            "loss_l1": self.loss_l1.item(),
            "loss_adv": self.loss_G_GAN.item(),
            "lambda_adv": self.lambda_adv,
            "d_loss": self.d_loss.item()
        }

    def update_lr(self):
        # There should be only one param_group.
        for p in self.optimizer_D.param_groups:
            current_lr = p['lr']
            update_lr = current_lr * 0.99
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.00006)
            p['lr'] = update_lr
            print("Decay net_D learning rate from %.6f to %.6f." % (current_lr, update_lr))

        for p in self.optimizer_G.param_groups:
            current_lr = p['lr']
            update_lr = current_lr * 0.99
            # minimum learning rate guarantee
            update_lr = max(update_lr, 0.0002)
            p['lr'] = update_lr
            print("Decay net_G learning rate from %.6f to %.6f." % (current_lr, update_lr))

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

    def save_networks(self, step):
        assert isinstance(self.netG.state_dict(), dict), "netG.state_dict() should be a dictionary"
        assert isinstance(self.netD.state_dict(), dict), "netD.state_dict() should be a dictionary"

        torch.save(self.netG.state_dict(), os.path.join(self.save_dir, f"{step}_net_G.pth"))
        torch.save(self.netD.state_dict(), os.path.join(self.save_dir, f"{step}_net_D.pth"))
        print(f" Checkpoint saved at step {step}")

    def load_networks(self, step):
        loaded = False
        target_filepath_G = os.path.join(self.save_dir, f"{step}_net_G.pth")
        target_filepath_D = os.path.join(self.save_dir, f"{step}_net_D.pth")

        # --- Generator ---
        if os.path.exists(target_filepath_G):
            loaded = True
            try:
                state_dict_G = torch.load(target_filepath_G, map_location=self.device)
                self.netG.load_state_dict(state_dict_G, strict=False)
                self._initialize_unmatched_weights(self.netG, state_dict_G, model_name="netG")
            except Exception as e:
                print(f" Error loading Generator: {e}")
        else:
            print(f" Generator checkpoint not found: {target_filepath_G}")

        # --- Discriminator ---
        if os.path.exists(target_filepath_D):
            try:
                state_dict_D = torch.load(target_filepath_D, map_location=self.device)
                self.netD.load_state_dict(state_dict_D, strict=False)
                self._initialize_unmatched_weights(self.netD, state_dict_D, model_name="netD")
            except Exception as e:
                print(f" Error loading Discriminator: {e}")
        else:
            print(f" Discriminator checkpoint not found: {target_filepath_D}")

        if loaded:
            print(f" Model {step} loaded successfully")
        return loaded

    def extract_keywords(self, name):
        KEYWORD_MATCH_RULES = ["down", "conv", "encoder", "decoder", "self", "line"]
        KEYWORD_MATCH_RULES.append("up")
        #KEYWORD_MATCH_RULES.append("res")
        return set([k for k in KEYWORD_MATCH_RULES if k in name])

    def extract_layer_name(self, name):
        parts = name.split('.')
        if parts:
            return parts[0]
        return name

    def _initialize_unmatched_weights(self, model, loaded_state_dict, model_name="Model"):
        model_state = model.state_dict()
        used_keys = set()

        shape_to_loaded_keys = {}
        name_to_layer = {}
        name_to_keywords = {}

        for k, v in loaded_state_dict.items():
            shape_to_loaded_keys.setdefault(v.shape, []).append(k)
            name_to_layer[k] = self.extract_layer_name(k)
            name_to_keywords[k] = self.extract_keywords(k)

        for name, param in model.named_parameters():
            full_name = name
            current_layer = self.extract_layer_name(full_name)
            current_keywords = self.extract_keywords(full_name)

            #print(f" Loading param (name - shape): {model_name}.{full_name} - {param.shape}")

            if full_name in loaded_state_dict and param.shape == loaded_state_dict[full_name].shape:
                param.data.copy_(loaded_state_dict[full_name])
                used_keys.add(full_name)
            else:
                matched = False
                candidate_keys = shape_to_loaded_keys.get(param.shape, [])
                for candidate in candidate_keys:
                    if candidate in used_keys:
                        continue

                    candidate_layer = name_to_layer.get(candidate)
                    candidate_keywords = name_to_keywords.get(candidate, set())

                    # 層級名稱與語意關鍵字需一致
                    if candidate_layer == current_layer and current_keywords & candidate_keywords:
                        print(f" Loading param (name - shape): {model_name}.{full_name} - {param.shape}")
                        print(f"  --> Shape & layer & keyword match. Copying from {candidate}")
                        param.data.copy_(loaded_state_dict[candidate])
                        used_keys.add(candidate)
                        matched = True
                        break

                if not matched:
                    print(f" Loading param (name - shape): {model_name}.{full_name} - {param.shape}")
                    print(f"  --> No suitable match found. Re-initializing param: {model_name}.{full_name}")
                    if "weight" in full_name:
                        # 暫時性的模型增加 conv.
                        init_smoothing_conv = False
                        #init_smoothing_conv = True
                        if init_smoothing_conv:
                            # 嘗試找對應的 bias
                            bias_name = name.replace("weight", "bias")
                            bias_param = model_state.get(bias_name, None)
                            matched = self.init_smoothing_conv_as_identity(param, bias_param)
                            if matched:
                                print(f" ✅  Initialized {model_name}.{name} as identity smoothing conv")
                                continue  # 跳過預設初始化

                        nn.init.kaiming_normal_(param.data, mode='fan_out', nonlinearity='leaky_relu')
                    elif "bias" in full_name:
                        nn.init.constant_(param.data, 0)

        for name, buffer in model.named_buffers():
            if name not in loaded_state_dict or model_state[name].shape != loaded_state_dict[name].shape:
                print(f" Re-initializing buffer (shape mismatch or missing): {model_name}.{name}")
                buffer.data.zero_()

    def init_smoothing_conv_as_identity(self, conv_param, bias_param=None):
        """將 smoothing conv 初始化為接近 identity（中心為 1，其餘為 0）"""
        if conv_param.shape[2:] == (3, 3) and conv_param.shape[0] == conv_param.shape[1]:
            with torch.no_grad():
                conv_param.zero_()
                c = conv_param.shape[0]
                for i in range(c):
                    conv_param[i, i, 1, 1] = 1.0  # 對角線中心位置為 1
                if bias_param is not None:
                    bias_param.zero_()
            return True
        return False

    def save_image(self, tensor: Union[torch.Tensor, List[torch.Tensor]]) -> np.ndarray:
        grid = vutils.make_grid(tensor)
        ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        return ndarr

    def anti_aliasing(self, image, strength=1):
        ksize = max(1, strength * 2 + 1)
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)
        return blurred

    def process_image(self, image, crop_src_font, canvas_size, resize_canvas, anti_aliasing_strength, binary_image):
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
        save_path = os.path.join(label_dir, f"{filename}.{image_ext}")
        if image_ext == "svg":
            save_path_pgm = os.path.join(label_dir, f"{filename}.pgm")
            cv2.imwrite(save_path_pgm, image)
            subprocess.call(['potrace', '-b', 'svg', '-u', '60', save_path_pgm, '-o', save_path])
            os.remove(save_path_pgm)
        else:
            cv2.imwrite(save_path, image)

    def sample(self, batch_data, basename, src_char_list=None, crop_src_font=False, canvas_size=256, resize_canvas=256,
               filename_rule="seq", binary_image=True, anti_aliasing_strength=1, image_ext="png"):
        with torch.no_grad():
            labels, image_B, image_A = batch_data
            model_input_data = {'label': labels, 'A': image_A, 'B': image_B}
            self.set_input(model_input_data)
            self.forward(model_input_data)

            output_images = torch.cat([self.fake_B, self.real_B], 3)
            for i, (label, image_tensor) in enumerate(zip(batch_data[0], output_images)):
                label_dir = os.path.join(basename, str(label.item()))
                os.makedirs(label_dir, exist_ok=True)

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
                    filename = str(i)

                opencv_image = cv2.cvtColor(self.save_image(image_tensor), cv2.COLOR_BGR2GRAY)
                processed_image = self.process_image(opencv_image, crop_src_font, canvas_size, resize_canvas,
                                                    anti_aliasing_strength, binary_image)
                self.save_image_to_disk(processed_image, label_dir, filename, image_ext)