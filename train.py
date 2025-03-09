#!/usr/bin/env python3
#encoding=utf-8
import argparse
import math
import os
import random
import sys
import time

import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader

from data import DatasetFromObj
from model import Zi2ZiModel


def chkormakedir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def empty_google_driver_trash(drive_service):
    if not drive_service is None:
        try:
          # 清空 google drive垃圾桶
          response = drive_service.files().emptyTrash().execute()
          #print("google drive垃圾桶已清空。")
        except Exception as e:
          print(f"發生錯誤：{e}")
    else:
        #print("drive_service is None")
        pass

def train(args):
    args = parser.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    data_dir = os.path.join(args.experiment_dir, "data")
    checkpoint_dir = os.path.join(args.experiment_dir, "checkpoint")
    chkormakedir(checkpoint_dir)

    # overwrite data dir path.
    if args.data_dir:
        data_dir = args.data_dir
        print("access data object at path: %s" % (data_dir))

    # overwrite checkpoint dir path.
    if args.checkpoint_dir :
        checkpoint_dir = args.checkpoint_dir
        chkormakedir(checkpoint_dir)
    print("access checkpoint object at path: %s" % (checkpoint_dir))

    self_attention=False
    if args.self_attention:
        self_attention=True
    residual_block=False
    if args.residual_block:
        residual_block=True

    drive_service = None
    if args.checkpoint_only_last:
        try:
            from google.colab import auth
            from googleapiclient.discovery import build
            # 1. 身份驗證
            auth.authenticate_user()
            # 2. 建立 Google Drive API 服務
            drive_service = build('drive', 'v3')            
        except Exception as e:
            print(f"發生錯誤：{e}")
            pass

    g_blur = False
    if args.g_blur:
        g_blur = True
    d_blur = False
    if args.d_blur:
        d_blur = True

    start_time = time.time()

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
        new_final_channels=args.new_final_channels,
        g_blur=g_blur,
        d_blur=d_blur,
        lr=args.lr
    )

    model.setup()
    model.print_networks(True)
    if args.resume:
        model.load_networks(args.resume)

    train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'),input_nc=args.input_nc)
    total_batches = math.ceil(len(train_dataset) / args.batch_size)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    global_steps = 0
    need_flush = False
    for epoch in range(args.epoch):
        for bid, batch in enumerate(dataloader):
            model.set_input(batch[0], batch[2], batch[1])
            const_loss, l1_loss, category_loss, cheat_loss = model.optimize_parameters()
            if bid % 100 == 0:
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %4.2f, d_loss: %.5f, g_loss: %.5f, " + \
                             "category_loss: %.5f, cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f"
                print(log_format % (epoch, bid, total_batches, passed, model.d_loss.item(), model.g_loss.item(),
                                    category_loss, cheat_loss, const_loss, l1_loss))
            if global_steps % args.checkpoint_steps == 0:
                if global_steps >= args.checkpoint_steps_after:
                    print("Checkpoint: checkpoint step %d" % global_steps)
                    model.save_networks(global_steps)
                    if args.checkpoint_only_last:
                        for checkpoint_index in range(0, global_steps, args.checkpoint_steps):
                            target_filepath = os.path.join(checkpoint_dir, str(checkpoint_index) + "_net_D.pth")
                            if os.path.isfile(target_filepath):
                                os.remove(target_filepath)
                            target_filepath = os.path.join(checkpoint_dir, str(checkpoint_index) + "_net_G.pth")
                            if os.path.isfile(target_filepath):
                                os.remove(target_filepath)
                        empty_google_driver_trash(drive_service)
                else:
                    print("Checkpoint: checkpoint step %d, will save after %d" % (global_steps, args.checkpoint_steps_after))
                need_flush = False
            else:
                need_flush = True
            global_steps += 1
        if (epoch + 1) % args.schedule == 0:
            model.update_lr()
    if need_flush:
        model.save_networks(global_steps)

    passed = time.time() - start_time
    print('passed time: %4.1f' % (passed))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--experiment_dir', required=True,
                        help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='overwrite data dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='overwrite checkpoint dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--image_size', type=int, default=256,
                        help="size of your input and output image")
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--embedding_num', type=int, default=40,
                        help="number for distinct embeddings")
    parser.add_argument('--embedding_dim', type=int, default=64, help="dimension for embedding")
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--schedule', type=int, default=20, help='number of epochs to half learning rate')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help="freeze encoder weights during training")
    parser.add_argument('--fine_tune', type=str, default=None,
                        help='specific labels id to be fine tuned')
    parser.add_argument('--inst_norm', action='store_true',
                        help='use conditional instance normalization in your model')
    parser.add_argument('--checkpoint_steps', type=int, default=100,
                        help='number of batches in between two checkpoints')
    parser.add_argument('--checkpoint_steps_after', type=int, default=1,
                        help='save the number of batches after')
    parser.add_argument('--checkpoint_only_last', action='store_true',
                        help='remove all previous versions, only keep last version')
    parser.add_argument('--flip_labels', action='store_true',
                        help='whether flip training data labels or not, in fine tuning')
    parser.add_argument('--random_seed', type=int, default=777,
                        help='random seed for random and pytorch')
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--input_nc', type=int, default=3,
                        help='number of input images channels')
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--residual_block', action='store_true')
    parser.add_argument('--sequence_count', type=int, default=9, help="discriminator layer count")
    parser.add_argument('--final_channels', type=int, default=1, help="discriminator final channels")
    parser.add_argument('--new_final_channels', type=int, default=0, help="new discriminator final channels")
    parser.add_argument('--g_blur', action='store_true')
    parser.add_argument('--d_blur', action='store_true')

    args = parser.parse_args()
    train(args)
