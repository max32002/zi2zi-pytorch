#!/usr/bin/env python3
#encoding=utf-8
import argparse
import collections
import json
import os
import pprint
import random
import re
import sys

import numpy as np
import torch
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from model import Zi2ZiModel
from utils.charset_util import processGlyphNames


def draw_single_char(ch, font, canvas_size):
    img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
    draw = ImageDraw.Draw(img)
    try:
        draw.text((10, 10), ch, 255, font=font)
    except OSError:
        return None
    bbox = img.getbbox()
    if bbox is None:
        return None
    l, u, r, d = bbox
    l = max(0, l - 5)
    u = max(0, u - 5)
    r = min(canvas_size * 2 - 1, r + 5)
    d = min(canvas_size * 2 - 1, d + 5)
    if l >= r or u >= d:
        return None
    img = np.array(img)
    img = img[u:d, l:r]
    img = 255 - img
    img = Image.fromarray(img)
    # img.show()
    width, height = img.size
    # Convert PIL.Image to FloatTensor, scale from 0 to 1, 0 = black, 1 = white
    try:
        img = transforms.ToTensor()(img)
    except SystemError:
        return None
    img = img.unsqueeze(0)  # 加轴
    pad_len = int(abs(width - height) / 2)  # 预填充区域的大小
    # 需要填充区域，如果宽大于高则上下填充，否则左右填充
    if width > height:
        fill_area = (0, 0, pad_len, pad_len)
    else:
        fill_area = (pad_len, pad_len, 0, 0)
    # 填充像素常值
    fill_value = 1
    img = nn.ConstantPad2d(fill_area, fill_value)(img)
    # img = nn.ZeroPad2d(m)(img) #直接填0
    img = img.squeeze(0)  # 去轴
    img = transforms.ToPILImage()(img)
    img = img.resize((canvas_size, canvas_size), Image.BILINEAR)
    return img


parser = argparse.ArgumentParser()

# create settings
parser.add_argument('--src_fonts_dir', type=str, default='charset/ZhongHuaSong', help='path of the src fonts')
parser.add_argument('--fonts_json', type=str, default=None, help='path of the target fonts\' json info.')
parser.add_argument('--char_size', type=int, default=250, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save samples')
parser.add_argument('--charset_path', type=str, default='charset/charset_xl.txt', help='path of charset file')
parser.add_argument('--bad_fonts', type=str, default='charset/error_font.txt', help='path of bad font list file')

# model settings
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
parser.add_argument('--image_size', type=int, default=256,
                    help="size of your input and output image")
parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
# parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                    help='weight for category loss')
parser.add_argument('--embedding_num', type=int, default=40,
                    help="number for distinct embeddings")
parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--input_nc', type=int, default=1)

args = parser.parse_args()


def get_fonts():
    dst_json = args.fonts_json
    with open(dst_json, 'r', encoding='utf-8') as fp:
        dst_fonts = json.load(fp)
    return dst_fonts


def chkormkdir(path):
    if os.path.isdir(path):
        return
    else:
        os.mkdir(path)
        return


class create_mix_ch_handle:

    def __init__(self, fonts, fake_prob=0.1):
        src_fonts_dir = args.src_fonts_dir
        fontPlane00 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'))
        fontPlane02 = TTFont(os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'))

        self.charSetPlane00 = processGlyphNames(fontPlane00.getGlyphNames())
        self.charSetPlane02 = processGlyphNames(fontPlane02.getGlyphNames())

        self.charSetTotal = self.charSetPlane00 | self.charSetPlane02
        self.charListTotal = list(self.charSetTotal)

        self.fontPlane00 = ImageFont.truetype(
            os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane00_2020051520200519101119.TTF'), args.char_size)
        self.fontPlane02 = ImageFont.truetype(
            os.path.join(src_fonts_dir, 'FZSONG_ZhongHuaSongPlane02_2020051520200519101142.TTF'), args.char_size)

        self.fonts = fonts
        with open(args.bad_fonts, 'r', encoding='utf-8') as bd_fs:
            self.bad_font_ids = [int(s) for s in bd_fs.readline().strip().split()]

        self.fake_prob = fake_prob

        checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")

        self.model = Zi2ZiModel(
            input_nc=args.input_nc,
            embedding_num=args.embedding_num,
            embedding_dim=args.embedding_dim,
            Lconst_penalty=args.Lconst_penalty,
            Lcategory_penalty=args.Lcategory_penalty,
            save_dir=checkpoint_dir,
            gpu_ids=args.gpu_ids,
            is_training=False
        )
        self.model.setup()
        self.model.print_networks(True)
        self.model.load_networks(args.resume)

    def set_cur_font(self, idx):
        self.idx = idx
        cur_font = self.fonts[idx]
        self.font_name = cur_font['font_name']
        print(self.font_name + ': ' + str(idx), flush=True)

        font_path = cur_font['font_pth']
        self.font_missing = set(cur_font['missing'])
        self.font_fake = set(cur_font['fake'])
        self.dst_font = ImageFont.truetype(font_path, args.char_size)
        self.dst_font_chars = set(processGlyphNames(TTFont(font_path).getGlyphNames()))

    def get_fake_single_char(self, ch):
        if ch in self.charSetPlane00:
            input_img = draw_single_char(ch, self.fontPlane00, args.canvas_size)
        elif ch in self.charSetPlane02:
            input_img = draw_single_char(ch, self.fontPlane02, args.canvas_size)
        else:
            return None
        input_img = input_img.convert('L')
        input_tensor = transforms.ToTensor()(input_img)
        input_tensor = transforms.Normalize(0.5, 0.5)(input_tensor).unsqueeze(0)
        label = torch.tensor(self.idx, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            self.model.set_input(label, input_tensor, input_tensor)
            self.model.forward()
            output_tensor = self.model.fake_B.detach().cpu().squeeze(dim=0)
            img = transforms.ToPILImage('L')(output_tensor)
            return img

    def get_mix_character(self, ch):
        # can draw this ch
        if ch in self.dst_font_chars and ch not in self.font_fake:
            if self.idx in self.bad_font_ids or random.random() > self.fake_prob:
                img = draw_single_char(ch, self.dst_font, args.canvas_size)
                return img, True
            else:
                img = self.get_fake_single_char(ch)
                return img, False
        # can't draw this ch
        else:
            # bad font, can't use font magic
            if self.idx in self.bad_font_ids:
                return None, True
            else:
                img = self.get_fake_single_char(ch)
                return img, False


if __name__ == "__main__":
    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)

    font_json_list = []
    with open(args.fonts_json, 'r', encoding='utf-8') as fp:
        for line in fp:
            font_json = json.loads(line.strip())
            font_json_list.append(font_json)

    mix_ch_handle = create_mix_ch_handle(font_json_list)
    sys.stdout.flush()

    with open(args.charset_path, 'r', encoding='utf-8') as fp:
        charset = [line.strip() for line in fp.readlines()]

    print('load charset, %d chars in total' % len(charset), flush=True)
    for ch in charset:
        '''
        if ch not in mix_ch_handle.charSetTotal:
            print('char %c is strange and will not be included.' % ch)
            print('encoding %s' % ch.encode('utf-8'))
        else:
            chkormkdir(os.path.join(args.sample_dir, ch))
        '''
        chkormkdir(os.path.join(args.sample_dir, ch))

    # Get start_num, font_cnt
    start_num = 0
    font_cnt = len(font_json_list)
    end_num = start_num + font_cnt

    for idx in range(start_num, end_num):
        mix_ch_handle.set_cur_font(idx)
        font_name = mix_ch_handle.font_name
        for ch in charset:
            save_path = os.path.join(args.sample_dir, ch)
            img, flag = mix_ch_handle.get_mix_character(ch)
            if img is None:
                continue
            else:
                if flag:
                    img.save(os.path.join(save_path, font_name + '_' + ch + '.png'))
                else:
                    img.save(os.path.join(save_path, font_name + '_' + ch + '_from_font_magic.png'))
