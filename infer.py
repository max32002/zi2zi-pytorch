import argparse
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image

from data import DatasetFromObj
from model import Zi2ZiModel


def ensure_dir(path):
    """確保目錄存在，不存在則建立"""
    os.makedirs(path, exist_ok=True)

def convert_to_gray_binary(image, ksize=1, threshold=127):
    """將 PIL 圖像轉換為灰階二值 OpenCV 圖像"""
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    blurred_image = cv2.GaussianBlur(opencv_image, (ksize, ksize), 0) if ksize > 0 else opencv_image
    _, binary_image = cv2.threshold(blurred_image, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

def draw_single_char(char, font, canvas_size, x_offset=0, y_offset=0):
    """繪製單個字元並返回灰階二值圖像"""
    image = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((x_offset, y_offset), char, (0, 0, 0), font=font)
    gray_image = image.convert('L')
    return convert_to_gray_binary(gray_image)

def load_char_list(filepath):
    """從檔案載入字元列表"""
    filepath = os.path.expanduser(filepath)
    if not os.path.exists(filepath):
        print(f"src_txt_file not found: {filepath}")
        return ""
    with open(filepath, 'r', encoding='utf-8') as file:
        return ''.join(line.strip() for line in file)

def create_dataloader(char_list, label, canvas_size, font, x_offset, y_offset, skip_exist, infer_dir, image_ext, filename_rule, ignore_int_array):
    """建立 DataLoader"""
    image_tensors, valid_chars = [], []
    label_list = []
    for char in char_list:
        image_filename = str(ord(char)) if filename_rule == "unicode_int" else ""
        save_path = os.path.join(infer_dir, image_filename + '.' + image_ext) if image_filename else ""
        if skip_exist and os.path.exists(save_path):
            continue
        if ord(char) in ignore_int_array:
            continue
        valid_chars.append(char)
        image = draw_single_char(char, font, canvas_size, x_offset, y_offset)
        image_tensors.append(transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(image)).unsqueeze(dim=0))
        label_list.append(label)

    if not image_tensors:
        return None, ""
    image_tensor = torch.cat(image_tensors, dim=0)
    label_tensor = torch.tensor(label_list)
    dataset = TensorDataset(label_tensor, image_tensor, image_tensor)
    return DataLoader(dataset, batch_size=len(valid_chars), shuffle=False), valid_chars

def infer(args):
    """推論主函數"""
    experiment_dir = args.experiment_dir
    checkpoint_dir = args.checkpoint_dir or os.path.join(experiment_dir, "checkpoint")
    infer_dir = os.path.expanduser(args.infer_dir) if args.infer_dir else os.path.join(experiment_dir, "infer")
    ensure_dir(infer_dir)
    ensure_dir(os.path.join(infer_dir, str(args.label)))
    print(f"Generate infer images at path: {infer_dir}")
    print(f"Access checkpoint object at path: {checkpoint_dir}")

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        image_size=args.image_size,
        is_training=False,
        self_attention=args.self_attention,
        d_spectral_norm=args.d_spectral_norm
    )
    model.print_networks(True)
    if not model.load_networks(args.resume):
        return

    char_list = args.src_txt if args.src_txt else load_char_list(args.src_txt_file)
    font = ImageFont.truetype(args.src_font, size=args.char_size)
    filename_rule = args.filename_rule
    ignore_int_array = [8, 10, 12, 32, 160, 4447, 8194]
    each_loop_length = args.batch_size
    total_rounds = (len(char_list) + args.batch_size - 1) // each_loop_length
    print(f"Total rounds: {total_rounds}")

    t0 = time.time()
    for current_round in range(total_rounds):
        round_chars = char_list[current_round * each_loop_length: (current_round + 1) * each_loop_length]
        print(f"Current round: {current_round + 1}/{total_rounds}")
        dataloader, valid_chars = create_dataloader(
            round_chars, args.label, args.canvas_size, font, args.src_font_x_offset, args.src_font_y_offset,
            args.skip_exist, os.path.join(infer_dir, str(args.label)), args.image_ext, filename_rule, ignore_int_array
        )
        if not dataloader:
            print("Image list is empty, skip this round")
            continue
        model.sample(
            next(iter(dataloader)), infer_dir, src_char_list=valid_chars, crop_src_font=args.crop_src_font,
            canvas_size=args.canvas_size, resize_canvas=args.resize_canvas,
            filename_rule=args.filename_rule, binary_image=True, anti_aliasing_strength=args.anti_alias,
            image_ext=args.image_ext
        )
    t_finish = time.time()
    print(f'Cold start time: {t_finish - t0:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('--anti_alias', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--canvas_size', type=int, default=256)
    parser.add_argument('--char_size', type=int, default=256)
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='overwrite checkpoint dir path')
    parser.add_argument('--crop_src_font', action='store_true')
    parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
    parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--filename_rule', type=str, default="unicode_int", choices=['seq', 'char', 'unicode_int', 'unicode_hex'])
    parser.add_argument('--from_txt', action='store_true')
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--image_ext', type=str, default='png', help='infer image format')
    parser.add_argument('--image_size', type=int, default=256, help="size of your input and output image")
    parser.add_argument('--infer_dir', type=str, default=None, help='overwrite infer dir path')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--resize_canvas', type=int, default=0)
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--run_all_label', action='store_true')
    parser.add_argument('--self_attention', action='store_true', help='use self attention in generator')
    parser.add_argument('--skip_exist', action='store_true')

    parser.add_argument('--src_font', type=str, default='')
    parser.add_argument('--src_font_x_offset', type=int, default=0)
    parser.add_argument('--src_font_y_offset', type=int, default=0)
    parser.add_argument('--src_txt', type=str, default='')
    parser.add_argument('--src_txt_file', type=str, default=None)

    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--d_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    args = parser.parse_args()
    infer(args)
