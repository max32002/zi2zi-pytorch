#!/usr/bin/env python3
#encoding=utf-8
import os
import sys

import argparse
import cv2
import numpy as np

from PIL import Image, ImageFont, ImageDraw
import json
import collections
import re
from fontTools.ttLib import TTFont
from tqdm import tqdm
import random

from torch import nn
from torchvision import transforms

from utils.charset_util import processGlyphNames

def draw_single_char(ch, font, canvas_size, x_offset=0, y_offset=0):
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

def convert_to_gray_binary(example_img, ksize=1, threshold=127):
    ksize = 0
    opencv_image = cv2.cvtColor(np.array(example_img), cv2.COLOR_RGB2BGR)
    blurred = None
    if ksize > 0:
        blurred = cv2.GaussianBlur(opencv_image, (ksize, ksize), 0)
    else:
        blurred = opencv_image
    ret, example_img = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # conver to gray
    example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)
    return example_img

def draw_checkpoint2font_example(ch, src_infer, dst_font, canvas_size, x_offset, y_offset, filter_hashes):
    dst_img = draw_single_char(ch, dst_font, canvas_size, x_offset, y_offset)
    # check the filter example in the hashes or not
    if dst_img is None:
        print("draw fail at char: %s" % (ch))
        return None
        
    dst_hash = hash(dst_img.tobytes())
    if dst_hash in filter_hashes:
        return None

    filename_mode = "unicode_int"
    image_filename = ""
    if filename_mode == "unicode_int":
        image_filename = str(ord(ch))

    src_img = None
    if len(image_filename) > 0:
        saved_image_path = os.path.join(src_infer, image_filename + '.png')
        #print("image_path", saved_image_path)
        if os.path.exists(saved_image_path):
            src_img = Image.open(saved_image_path)
        else:
            print("path not exsit:", saved_image_path)
    
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    if src_img:
        example_img.paste(src_img, (canvas_size, 0))
    
    # convert to gray img
    #example_img = example_img.convert('L')

    example_img = convert_to_gray_binary(example_img, 1, 127)

    return example_img


def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    """ Some characters are missing in a given font, filter them
    by checking the recurring hashes
    """
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for c in sample:
        img = draw_single_char(c, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]


def checkpoint2font(src_infer, dst, charset, char_size, canvas_size,
             x_offset, y_offset, sample_count, sample_dir, label=0, filter_by_hash=True):
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0

    for c in charset:
        if count == sample_count:
            break
        e = draw_checkpoint2font_example(c, src_infer, dst_font, canvas_size, x_offset, y_offset, filter_hashes)
        if not e is None:        
            target_path = os.path.join(sample_dir, "%d_%05d.png" % (label, count))
            #e.save(target_path)
            cv2.imwrite(target_path, e)

            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['checkpoint2font'], required=True,
                    help='generate mode.\n'
                         'use --src_checkpoint_folder and --dst_font for checkpoint2font mode.\n'
                    )
parser.add_argument('--src_infer', type=str, default=None, help='path of the source infer image path')
parser.add_argument('--src_imgs', type=str, default=None, help='path of the source imgs')
parser.add_argument('--dst_font', type=str, default=None, help='path of the target font')
parser.add_argument('--dst_imgs', type=str, default=None, help='path of the target imgs')

parser.add_argument('--filter', default=False, action='store_true', help='filter recurring characters')
parser.add_argument('--charset', type=str, help='charset, a one line file.')
parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
parser.add_argument('--char_size', type=int, default=256, help='character size')
parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--x_offset', type=int, default=0, help='x offset')
parser.add_argument('--y_offset', type=int, default=0, help='y_offset')
parser.add_argument('--sample_count', type=int, default=5000, help='number of characters to draw')
parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')

args = parser.parse_args()

if __name__ == "__main__":
    input_img_path = os.path.abspath(args.src_infer)

    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    if args.mode == 'checkpoint2font':
        if args.src_infer is None or args.dst_font is None:
            raise ValueError('src_infer and dst_font are required.')
        if args.charset is None:
            raise ValueError('charset file are required.')
        charset = list(open(args.charset, encoding='utf-8').readline().strip())
        else:
            # auto
            if len(charset) == 0:
                target_folder_list = os.listdir(input_img_path)
                for item in target_folder_list:
                    if item.endswith(".png"):
                        #print("image file name", item)
                        char_string = item.replace(".png","")
                        charset.append(chr(int(char_string)))

        if args.shuffle:
            np.random.shuffle(charset)
        checkpoint2font(args.src_infer, args.dst_font, charset, args.char_size,
                  args.canvas_size, args.x_offset, args.y_offset,
                  args.sample_count, args.sample_dir, args.label, args.filter)
    else:
        raise ValueError('mode should be font2font, font2imgs or imgs2imgs')
