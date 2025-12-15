#!/usr/bin/env python3
#encoding=utf-8
import argparse
import collections
import json
import os
import random
import re
import sys

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from utils.charset_util import processGlyphNames


def draw_character(char, font, canvas_size, x_offset=0, y_offset=0, auto_fit=True):
    """渲染單個字元到圖像。"""
    img = None
    if auto_fit:
        img = Image.new("L", (canvas_size * 2, canvas_size * 2), 0)
        draw = ImageDraw.Draw(img)
        draw.text((x_offset, y_offset), char, 255, font=font)

        bbox = img.getbbox()
        if bbox is None:
            print(f"警告：字型 '{font.path}' 中缺少字元 '{char}' 的 glyph。")
            return None
        l, u, r, d = bbox
        l = max(0, l - 5)
        u = max(0, u - 5)
        r = min(canvas_size * 2 - 1, r + 5)
        d = min(canvas_size * 2 - 1, d + 5)
        if l >= r or u >= d:
            print(f"警告：字型 '{font.path}' 中缺少字元 '{char}' 的 glyph。")
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
        except Exception as e:
            print(f"Error ToTensor: {e}")
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
        img = img.squeeze(0)
        img = transforms.ToPILImage()(img)
        img = img.resize((canvas_size, canvas_size), Image.BILINEAR)
    else:
        img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((0 + x_offset, 0 + y_offset), char, (0, 0, 0), font=font)
        img = img.convert('L')
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

def draw_font2font_example(char, src_font, dst_font, canvas_size, src_x_offset, src_y_offset, dst_x_offset, dst_y_offset, filter_hashes, auto_fit=True):
    target_image = draw_character(char, dst_font, canvas_size, dst_x_offset, dst_y_offset, auto_fit=auto_fit)
    if target_image is None:
        print(f"渲染字元失敗：{char}")
        return None

    dst_hash = hash(target_image.tobytes())
    if dst_hash in filter_hashes:
        return None
    src_img = draw_character(char, src_font, canvas_size, src_x_offset, src_y_offset, auto_fit=auto_fit)
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(target_image, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    example_img = convert_to_gray_binary(example_img, 1, 127)
    return example_img

def draw_font2imgs_example(char, src_font, canvas_size, x_offset, y_offset,auto_fit=True):
    src_img = draw_character(char, src_font, canvas_size, x_offset, y_offset, auto_fit=auto_fit)
    example_img = convert_to_gray_binary(src_img, 0, 127)
    return example_img

def draw_imgs2imgs_example(src_img, dst_img, canvas_size):
    src_img = src_img.resize((canvas_size, canvas_size), Image.BILINEAR).convert('RGB')
    dst_img = dst_img.resize((canvas_size, canvas_size), Image.BILINEAR).convert('RGB')
    example_img = Image.new("RGB", (canvas_size * 2, canvas_size), (255, 255, 255))
    example_img.paste(dst_img, (0, 0))
    example_img.paste(src_img, (canvas_size, 0))
    # convert to gray img
    example_img = example_img.convert('L')
    return example_img

def filter_recurring_hash(charset, font, canvas_size, x_offset, y_offset):
    _charset = charset.copy()
    np.random.shuffle(_charset)
    sample = _charset[:2000]
    hash_count = collections.defaultdict(int)
    for ch in sample:
        img = draw_character(ch, font, canvas_size, x_offset, y_offset)
        hash_count[hash(img.tobytes())] += 1
    recurring_hashes = filter(lambda d: d[1] > 2, hash_count.items())
    return [rh[0] for rh in recurring_hashes]

def font2font(src, dst, charset, char_size, canvas_size,
             src_x_offset, src_y_offset, dst_x_offset, dst_y_offset,
             sample_dir, label=0, filter_by_hash=True, auto_fit=True):
    src_font = ImageFont.truetype(src, size=char_size)
    dst_font = ImageFont.truetype(dst, size=char_size)

    filter_hashes = set()
    if filter_by_hash:
        filter_hashes = set(filter_recurring_hash(charset, dst_font, canvas_size, x_offset, y_offset))
        print("filter hashes -> %s" % (",".join([str(h) for h in filter_hashes])))

    count = 0
    for char in charset:
        e = draw_font2font_example(char, src_font, dst_font, canvas_size, src_x_offset, src_y_offset, dst_x_offset, dst_y_offset, filter_hashes, auto_fit=auto_fit)
        if not e is None:
            target_path = os.path.join(sample_dir, "%d_%05d.png" % (label, count))
            #e.save(target_path)
            cv2.imwrite(target_path, e)
            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)


def font2imgs(src, charset, char_size, canvas_size, x_offset, y_offset, sample_dir, label=0, auto_fit=True,
    filename_with_label=True, filename_rule="seq", enable_txt=False, caption_text = ""):
    src_font = ImageFont.truetype(src, size=char_size)
    count = 0
    for char in charset:
        e = draw_font2imgs_example(char, src_font, canvas_size, x_offset, y_offset, auto_fit)
        if not e is None:
            filename_prefix = "%d_" % (label)
            if not filename_with_label:
                filename_prefix = ""
            filename = "%05d" % (count)
            if filename_rule=="unicode_int":
                filename = f"{ord(char)}"
            if filename_rule=="unicode_hex":
                filename = f"{ord(char):x}"
            target_filename = filename_prefix + filename + ".png"
            target_path = os.path.join(sample_dir, target_filename)
            #e.save(target_path)
            cv2.imwrite(target_path, e)

            if enable_txt:
                text_filename = filename_prefix + filename + ".txt"
                text_path = os.path.join(sample_dir, text_filename)
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(f"{caption_text} {char}")

            count += 1
            if count % 500 == 0:
                print("processed %d chars" % count)

def imgs2imgs(src, dst, canvas_size, sample_dir):
    label_map = {
        '1号字体': 0,
        '2号字体': 1,
    }
    count = 0
    # We only need character in source img.
    source_pattern = re.compile('(.)~0号字体')
    # We need character and label in target img.
    target_pattern = re.compile('(.)~(/d)号字体')

    # Multi-imgs with a same character in src_imgs are allowed.
    # Use default_dict(list) to storage.
    source_ch_list = collections.defaultdict(list)
    for c in tqdm(os.listdir(src)):
        res = re.match(source_pattern, c)
        ch = res[1]
        source_ch_list[ch].append(c)

    def get_source_img(ch):
        res = source_ch_list.get(ch)
        if res is None or len(res) == 0:
            return None
        if len(res) == 1:
            return res[0]
        idx = random.randint(0, len(res))
        return res[idx]

    for c in tqdm(os.listdir(dst)):
        res = re.match(target_pattern, c)
        ch = res[1]
        label = label_map[res[2]]
        src_img_name = get_source_img(ch)
        if src_img_name is None:
            continue
        img_path = os.path.join(src, src_img_name)
        src_img = Image.open(img_path)
        img_path = os.path.join(dst, c)
        dst_img = Image.open(img_path)
        e = draw_imgs2imgs_example(src_img, dst_img, canvas_size)
        if e:
            e.save(os.path.join(sample_dir, "%d_%05d.png" % (label, count)))
            count += 1

def font2imge(args):
    auto_fit = True
    if args.disable_auto_fit:
        auto_fit = False

    filename_with_label = True
    if args.disable_filename_label:
        filename_with_label = False

    if not os.path.isdir(args.sample_dir):
        os.mkdir(args.sample_dir)
    if args.mode == 'font2font':
        if args.src_font is None or args.dst_font is None:
            raise ValueError('src_font and dst_font are required.')
        if args.charset is None:
            raise ValueError('charset file are required.')
        charset = list(open(args.charset, encoding='utf-8').readline().strip())
        if args.shuffle:
            np.random.shuffle(charset)
        font2font(args.src_font, args.dst_font, charset, args.char_size,
                  args.canvas_size, args.src_x_offset, args.src_y_offset, args.dst_x_offset, args.dst_y_offset,
                  args.sample_dir, args.label, args.filter, auto_fit=auto_fit)
    elif args.mode == 'font2imgs':
        if args.src_font is None:
            raise ValueError('src_font and dst_font are required.')
        if args.charset is None:
            raise ValueError('charset file are required.')
        charset = list(open(args.charset, encoding='utf-8').readline().strip())
        if args.shuffle:
            np.random.shuffle(charset)
        font2imgs(args.src_font, charset, args.char_size,
                  args.canvas_size, args.src_x_offset, args.src_y_offset,
                  args.sample_dir, auto_fit=auto_fit,
                  filename_with_label=filename_with_label, filename_rule=args.filename_rule,
                  enable_txt=args.enable_txt, caption_text=args.caption_text)
    else:
        raise ValueError('mode should be font2font, font2imgs or imgs2imgs')

if __name__ == "__main__":
    random.seed()
    np.random.seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['imgs2imgs', 'font2imgs', 'font2font'], required=True,
                        help='generate mode.\n'
                             'use --src_imgs and --dst_imgs for imgs2imgs mode.\n'
                             'use --src_font and --dst_imgs for font2imgs mode.\n'
                             'use --src_font and --dst_font for font2font mode.\n'
                             'No imgs2font mode.'
                        )
    parser.add_argument('--canvas_size', type=int, default=256, help='canvas size')
    parser.add_argument('--caption_text', type=str, default="")
    parser.add_argument('--char_size', type=int, default=256, help='character size')
    parser.add_argument('--charset', type=str, help='one line file.')
    parser.add_argument('--disable_auto_fit', action='store_true', help='disable image auto fit')
    parser.add_argument('--disable_filename_label', action='store_true', help='disable image filename with label')
    parser.add_argument('--dst_font', type=str, default=None, help='path of the target font')
    parser.add_argument('--dst_imgs', type=str, default=None, help='path of the target imgs')
    parser.add_argument('--dst_x_offset', type=int, default=0, help='x offset')
    parser.add_argument('--dst_y_offset', type=int, default=0, help='y_offset')
    parser.add_argument('--enable_txt', action='store_true', help='store image caption to text file')
    parser.add_argument('--filename_rule', type=str, default="seq", choices=['seq', 'char', 'unicode_int', 'unicode_hex'])
    parser.add_argument('--filter', default=False, action='store_true', help='filter recurring characters')
    parser.add_argument('--label', type=int, default=0, help='label as the prefix of examples')
    parser.add_argument('--sample_dir', type=str, default='sample_dir', help='directory to save examples')
    parser.add_argument('--shuffle', default=False, action='store_true', help='shuffle a charset before processings')
    parser.add_argument('--src_font', type=str, default=None, help='path of the source font')
    parser.add_argument('--src_fonts_dir', type=str, default=None, help='path of the source fonts')
    parser.add_argument('--src_imgs', type=str, default=None, help='path of the source imgs')
    parser.add_argument('--src_x_offset', type=int, default=0, help='x offset')
    parser.add_argument('--src_y_offset', type=int, default=0, help='y_offset')
    
    args = parser.parse_args()
    font2imge(args)