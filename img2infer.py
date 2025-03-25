#!/usr/bin/env python3
#encoding=utf-8
import argparse
import glob
import math
import os
import random
import time
from os.path import expanduser

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

def infer(args):
    """推論主函數"""
    experiment_dir = Path(args.experiment_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir or experiment_dir / "checkpoint").resolve()
    infer_dir = Path(os.path.expanduser(args.experiment_infer_dir)) if args.experiment_infer_dir else experiment_dir / "infer"
    infer_dir.mkdir(parents=True, exist_ok=True)
    (infer_dir / str(args.label)).mkdir(parents=True, exist_ok=True)
    input_img_path = Path(args.src_infer).resolve()

    print(f"Generate infer images at path: {infer_dir}")
    print(f"Access checkpoint object at path: {checkpoint_dir}")
    print(f"source images from path: {input_img_path}")

    model = Zi2ZiModel(
        input_nc=args.input_nc, embedding_num=args.embedding_num, embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty, Lcategory_penalty=args.Lcategory_penalty, save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids, self_attention=args.self_attention,
        residual_block=args.residual_block, is_training=False
    )
    model.setup()
    model.print_networks(True)
    if not model.load_networks(args.resume):
        print("Error: Failed to load model networks.")
        return

    t1 = time.time()

    target_folder_list = os.listdir(input_img_path)
    char_array = []
    for filename in target_folder_list:
        if filename.endswith(".png"):
            #print("image file name", filename)
            char_string = os.path.splitext(filename)[0]
            char_array.append(chr(int(char_string)))
    src_char_list = ''.join(char_array)

    final_batch_size = args.batch_size
    total_length = 0
    total_length = len(src_char_list)

    each_loop_length = args.each_loop_length
    total_round = int(total_length/each_loop_length) + 1

    if total_round > 1:
        print("Total round: %d" % (total_round))

    for current_round in range(total_round):
        if total_round > 1:
            print(f"Current round: {current_round + 1}")

        current_round_text_excepted = src_char_list[current_round * each_loop_length: (current_round + 1) * each_loop_length]
        current_round_text_real = ""
        img_list = []

        if total_round > 1:
            print(f"Start to draw char at round: {current_round + 1}/{total_round}")

        for ch in current_round_text_excepted:
            image_filename = str(ord(ch))
            input_image_path = input_img_path / f"{image_filename}.{args.image_ext}"
            try:
                src_img = Image.open(input_image_path).convert('L')
                img_list.append(transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(src_img)).unsqueeze(dim=0))
                current_round_text_real += ch
            except FileNotFoundError:
                print(f"Warning: Image path not found: {input_image_path}")

        if not img_list:
            continue

        if total_round > 1:
            print(f"Start to infer char at round: {current_round + 1}/{total_round}")

        img_list = torch.cat(img_list, dim=0)
        label_list = torch.tensor([args.label] * len(img_list))
        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=len(img_list), shuffle=False)
        current_round_text = current_round_text_real

        for batch in dataloader:
            model.sample(
                batch, infer_dir, src_char_list=current_round_text, crop_src_font=args.crop_src_font,
                canvas_size=args.canvas_size, resize_canvas=args.resize_canvas,
                filename_rule=args.filename_rule, binary_image=True, anti_aliasing_strength=args.anti_alias,
                image_ext=args.image_ext
            )

    t_finish = time.time()
    print(f'Cold start time: {t_finish - t0:.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('--anti_alias', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--canvas_size', type=int, default=256)
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='overwrite checkpoint dir path')
    parser.add_argument('--crop_src_font', action='store_true')
    parser.add_argument('--each_loop_length', type=int, default=200)
    parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
    parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--experiment_infer_dir', type=str, default=None, help='overwrite infer dir path')
    parser.add_argument('--filename_rule', type=str, default="unicode_int", choices=['seq', 'char', 'unicode_int', 'unicode_hex'])
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--image_ext', type=str, default='png', help='infer image format')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--residual_block', action='store_true')
    parser.add_argument('--resize_canvas', type=int, default=0)
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--src_infer', type=str, default='experiments/infer/0')
    parser.add_argument('--src_txt', type=str, default='')
    parser.add_argument('--src_txt_file', type=str, default=None)
    args = parser.parse_args()
    infer(args)
