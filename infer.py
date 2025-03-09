#!/usr/bin/env python3
#encoding=utf-8
import argparse
import math
import os
import random
import time
from os.path import expanduser

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import make_grid, save_image

from data import DatasetFromObj
from model import Zi2ZiModel
from model.model import chk_mkdir


def convert_to_gray_binary(example_img, ksize=1, threshold=127):
    opencvImage = cv2.cvtColor(np.array(example_img), cv2.COLOR_RGB2BGR)
    blurred = None
    if ksize > 0:
        blurred = cv2.GaussianBlur(opencvImage, (ksize, ksize), 0)
    else:
        blurred = opencvImage
    ret, example_img = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    # conver to gray
    example_img = cv2.cvtColor(example_img, cv2.COLOR_BGR2GRAY)
    return example_img

def draw_single_char(ch, font, canvas_size, x_offset = 0, y_offset = 0):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0 + x_offset, 0 + y_offset), ch, (0, 0, 0), font=font)
    img = img.convert('L')

    img = convert_to_gray_binary(img, 0, 127)
    return img

def infer(args):
    args = parser.parse_args()
    chk_mkdir(args.experiment_dir)
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    infer_dir = os.path.join(args.experiment_dir, "infer")

    # overwrite checkpoint dir path.
    if args.infer_dir :
        infer_dir = args.infer_dir
        if(infer_dir[:2]=='~/'):
            infer_dir = os.path.expanduser(infer_dir)
    
    print("generate infer images at path: %s" % (infer_dir))
    chk_mkdir(infer_dir)

    infer_with_label_dir = os.path.join(infer_dir, str(args.label))
    chk_mkdir(infer_with_label_dir)

    # overwrite checkpoint dir path.
    if args.checkpoint_dir :
        checkpoint_dir = args.checkpoint_dir
        print("access checkpoint object at path: %s" % (checkpoint_dir))

    self_attention=False
    if args.self_attention:
        self_attention=True
    residual_block=False
    if args.residual_block:
        residual_block=True

    t0 = time.time()

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        image_size=args.image_size,
        self_attention=self_attention,
        residual_block=residual_block,
        sequence_count=args.sequence_count,
        final_channels=args.final_channels,
        is_training=False
    )
    model.setup()
    model.print_networks(True)
    model.load_networks(args.resume)

    t1 = time.time()

    src_char_list = args.src_txt
    text_filepath = args.src_txt_file
    if text_filepath:
        is_file_exist = False
        if(text_filepath[:2]=='~/'):
            text_filepath = os.path.expanduser(text_filepath)
        is_file_exist = os.path.exists(text_filepath)

        if is_file_exist:
            src_char_list = ""
            char_array = []
            with open(text_filepath, 'r', encoding='utf-8') as fp:
                for s in fp.readlines():
                    char_array.append(s.strip())
            src_char_list = ''.join(char_array)
        else:
            print("src_txt_file not fould: %s" % (text_filepath))

    final_batch_size = args.batch_size

    total_length = 0
    if args.from_txt:
        total_length = len(src_char_list)

    each_loop_length = args.each_loop_length

    total_round = int(total_length/each_loop_length) + 1

    if total_round > 1:
        print("Total round: %d" % (total_round))
    
    font = ImageFont.truetype(args.src_font, size=args.char_size)
    filename_mode = "unicode_int"
    
    for current_round in range(total_round):
        if total_round > 1:
            print("Current round: %d" % (current_round+1))

        current_round_text = ""

        dataloader = None

        if args.from_txt:
            current_round_text_excepted = src_char_list[current_round*each_loop_length:(current_round+1)*each_loop_length]
            current_round_text_real = ""

            if total_round > 1:
                print("Start to draw char at round: %d/%d" % (current_round+1,total_round))
            
            img_list_array = []
            ignore_int_array = [8,10,12,32,160,4447,8194]
            for ch in current_round_text_excepted:
                saved_image_exist = False

                image_filename = ""
                if filename_mode == "unicode_int":
                    image_filename = str(ord(ch))

                if len(image_filename) > 0:
                    saved_image_path = os.path.join(infer_with_label_dir, image_filename + '.' + args.image_ext)
                    #print("ch:", ch, "image_path", saved_image_path)
                    if os.path.exists(saved_image_path):
                        saved_image_exist = True

                append_image = True

                if args.skip_exist:
                    if saved_image_exist:
                        append_image = False

                if ord(ch) in ignore_int_array:
                    append_image = False
                
                if append_image:
                    current_round_text_real += ch

                    img = draw_single_char(ch, font, args.canvas_size, args.src_font_x_offset, args.src_font_y_offset)
                    img_list_array.append(transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(
                        img
                    )).unsqueeze(dim=0))
            label_list = [args.label for _ in img_list_array]
            if total_round > 1:
                print("Start to infer char at round: %d/%d" % (current_round+1,total_round))

            current_round_length = len(current_round_text_real)
            #print("current_round_length", current_round_length)
            if final_batch_size < current_round_length:
                final_batch_size = current_round_length
            if current_round_length > 0:
                current_round_text = current_round_text_real
            else:
                print("img_list_array is empty, skip this round")
                continue

            img_list = torch.cat(img_list_array, dim=0)
            label_list = torch.tensor(label_list)

            dataset = TensorDataset(label_list, img_list, img_list)
            dataloader = DataLoader(dataset, batch_size=final_batch_size, shuffle=False)

        for batch in dataloader:
            resize_canvas_size = args.canvas_size
            if args.resize_canvas_size > 0:
                resize_canvas_size = args.resize_canvas_size
            model.sample(batch, infer_dir, src_char_list=current_round_text, crop_src_font=args.crop_src_font, canvas_size=args.canvas_size, resize_canvas_size = args.resize_canvas_size, filename_mode=args.filename_rule, binary_image=True, strength=args.anti_alias, image_ext=args.image_ext)

    t_finish = time.time()
    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('--experiment_dir', required=True,
                        help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='overwrite checkpoint dir path')
    parser.add_argument('--infer_dir', type=str, default=None,
                        help='overwrite infer dir path')
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--image_size', type=int, default=256,
                        help="size of your input and output image")
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    # parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
    parser.add_argument('--embedding_dim', type=int, default=64, help="dimension for embedding")
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
    parser.add_argument('--input_nc', type=int, default=1)
    parser.add_argument('--from_txt', action='store_true')
    parser.add_argument('--src_txt', type=str, default='')
    parser.add_argument('--src_txt_file', type=str, default=None)
    parser.add_argument('--canvas_size', type=int, default=256)
    parser.add_argument('--char_size', type=int, default=256)
    parser.add_argument('--label', type=int, default=0)
    parser.add_argument('--src_font', type=str, default='')
    parser.add_argument('--crop_src_font', action='store_true')
    parser.add_argument('--resize_canvas_size', type=int, default=0)
    parser.add_argument('--src_font_x_offset', type=int, default=0)
    parser.add_argument('--src_font_y_offset', type=int, default=0)
    parser.add_argument('--each_loop_length', type=int, default=32)
    parser.add_argument('--skip_exist', action='store_true')
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--residual_block', action='store_true')
    parser.add_argument('--anti_alias', type=int, default=0)
    parser.add_argument('--image_ext', type=str, default='png', help='infer image format')
    parser.add_argument('--sequence_count', type=int, default=9, help="discriminator layer count")
    parser.add_argument('--final_channels', type=int, default=1, help="discriminator final channels")
    parser.add_argument('--filename_rule', type=str, default="unicode_int", choices=['seq', 'char', 'unicode_int', 'unicode_hex'])

    args = parser.parse_args()
    infer(args)
