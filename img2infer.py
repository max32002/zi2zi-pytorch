#!/usr/bin/env python3
#encoding=utf-8
from data import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from model import Zi2ZiModel
import os
from os.path import expanduser
import argparse
import torch
import random
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import time
from model.model import chk_mkdir

parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('--experiment_dir', required=True,
                    help='experiment directory, data, samples,checkpoints,etc')
parser.add_argument('--experiment_checkpoint_dir', type=str, default=None,
                    help='overwrite checkpoint dir path')
parser.add_argument('--experiment_infer_dir', type=str, default=None,
                    help='overwrite infer dir path')
parser.add_argument('--start_from', type=int, default=0)
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
parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
parser.add_argument('--obj_path', type=str, default='./experiment/data/val.obj', help='the obj file you infer')
parser.add_argument('--input_nc', type=int, default=1)

parser.add_argument('--from_txt', action='store_true')
parser.add_argument('--generate_filename_mode', type=str, choices=['seq', 'char', 'unicode_hex', 'unicode_int'], 
                    help='generate filename mode.\n'
                         'use seq for sequence.\n'
                         'use char for character.\n'
                         'use unicode_hex for unicode hex .\n'
                         'use unicode_hex for unicode decimal.',
                    default="seq",
                    )
parser.add_argument('--src_txt', type=str, default='')
parser.add_argument('--src_txt_file', type=str, default=None)
parser.add_argument('--canvas_size', type=int, default=256)
parser.add_argument('--char_size', type=int, default=256)
parser.add_argument('--run_all_label', action='store_true')
parser.add_argument('--label', type=int, default=0)
parser.add_argument('--src_infer', type=str, default='experiments/infer/0')
parser.add_argument('--crop_src_font', action='store_true')
parser.add_argument('--resize_canvas_size', type=int, default=0)
parser.add_argument('--each_loop_length', type=int, default=200)
parser.add_argument('--self_attention', action='store_true')
parser.add_argument('--self_attention_layer', type=int, default=4, help="self attention append to layer")
parser.add_argument('--residual_block', action='store_true')
parser.add_argument('--residual_block_layer', nargs='*', default=[3,5], help="residual block append to layer (feature not work now)")
parser.add_argument('--anti_alias', type=int, default=1)
parser.add_argument('--image_ext', type=str, default='png', help='infer image format')

def main():
    args = parser.parse_args()
    chk_mkdir(args.experiment_dir)
    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    sample_dir = os.path.join(args.experiment_dir, "sample")
    infer_dir = os.path.join(args.experiment_dir, "infer")
    input_img_path = os.path.abspath(args.src_infer)

    # overwrite checkpoint dir path.
    if args.experiment_infer_dir :
        infer_dir = args.experiment_infer_dir
        if(infer_dir[:2]=='~/'):
            infer_dir = os.path.expanduser(infer_dir)
    
    print("generate infer images at path: %s" % (infer_dir))
    chk_mkdir(infer_dir)

    infer_with_label_dir = os.path.join(infer_dir, str(args.label))
    chk_mkdir(infer_with_label_dir)

    # overwrite checkpoint dir path.
    if args.experiment_checkpoint_dir :
        checkpoint_dir = args.experiment_checkpoint_dir
        print("access checkpoint object at path: %s" % (checkpoint_dir))

    self_attention=False
    if args.self_attention:
        self_attention=True
    self_attention_layer=args.self_attention_layer
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
        self_attention=self_attention,
        self_attention_layer=self_attention_layer,
        residual_block=residual_block,
        residual_block_layer=args.residual_block_layer,
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

    if src_char_list == "":
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
    if args.from_txt:
        total_length = len(src_char_list)

    each_loop_length = args.each_loop_length
    total_round = int(total_length/each_loop_length) + 1

    if total_round > 1:
        print("Total round: %d" % (total_round))

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
            img_list = []
            for ch in current_round_text_excepted:
                image_filename = ""
                if filename_mode == "unicode_int":
                    image_filename = str(ord(ch))

                src_img = None
                if len(image_filename) > 0:
                    input_image_path = os.path.join(input_img_path, image_filename + '.' + args.image_ext)
                    #print("ch:", ch, "image_path", input_image_path)
                    if os.path.exists(input_image_path):
                        src_img = Image.open(input_image_path)
                        src_img = src_img.convert('L')
                    else:
                        print("image path not exsit:", input_image_path)
                    
                if src_img:
                    current_round_text_real += ch
                    img_list.append(transforms.Normalize(0.5, 0.5)(transforms.ToTensor()(
                        src_img
                    )).unsqueeze(dim=0))
            label_list = [args.label for _ in img_list]
            if total_round > 1:
                print("Start to infer char at round: %d/%d" % (current_round+1,total_round))

            current_round_length = len(current_round_text_real)
            if final_batch_size < current_round_length:
                final_batch_size = current_round_length
            if current_round_length > 0:
                current_round_text = current_round_text_real
            else:
                continue

            img_list = torch.cat(img_list, dim=0)
            label_list = torch.tensor(label_list)

            dataset = TensorDataset(label_list, img_list, img_list)
            dataloader = DataLoader(dataset, batch_size=final_batch_size, shuffle=False)

        else:
            val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'),
                                         input_nc=args.input_nc,
                                         start_from=args.start_from)
            dataloader = DataLoader(val_dataset, batch_size=final_batch_size, shuffle=False)

        for batch in dataloader:
            if args.run_all_label:
                pass
            else:
                # model.set_input(batch[0], batch[2], batch[1])
                # model.optimize_parameters()
                resize_canvas_size = args.canvas_size
                if args.resize_canvas_size > 0:
                    resize_canvas_size = args.resize_canvas_size
                model.sample(batch, infer_dir, src_char_list=current_round_text, crop_src_font=args.crop_src_font, canvas_size=args.canvas_size, resize_canvas_size = args.resize_canvas_size, filename_mode=args.generate_filename_mode, binary_image=True, strength=args.anti_alias, image_ext=args.image_ext)
                print("done sample, goto next round")

        del dataloader
        torch.cuda.empty_cache()

    t_finish = time.time()
    print('cold start time: %.2f, hot start time %.2f' % (t_finish - t0, t_finish - t1))


if __name__ == '__main__':
    with torch.no_grad():
        main()
