import argparse
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

from data import DatasetFromObj
from model import Zi2ZiModel


def ensure_dir(path):
    """確保目錄存在，不存在則建立"""
    os.makedirs(path, exist_ok=True)

def clear_google_drive_trash(drive_service):
    """清空 Google Drive 垃圾桶"""
    if drive_service:
        try:
            drive_service.files().emptyTrash().execute()
            # print("Google Drive 垃圾桶已清空。")
        except Exception as e:
            print(f"清空 Google Drive 垃圾桶時發生錯誤：{e}")

def setup_google_drive_service():
    """設定 Google Drive 服務"""
    try:
        from google.colab import auth
        from googleapiclient.discovery import build

        auth.authenticate_user()
        return build('drive', 'v3')
    except ImportError:
        print("未檢測到 Google Colab 環境，無法設定 Google Drive 服務。")
        return None
    except Exception as e:
        print(f"設定 Google Drive 服務時發生錯誤：{e}")
        return None

def main():
    args = parser.parse_args()
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    data_dir = args.data_dir or os.path.join(args.experiment_dir, "data")
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.experiment_dir, "checkpoint")

    ensure_dir(checkpoint_dir)

    print(f"資料目錄：{data_dir}")
    print(f"檢查點目錄：{checkpoint_dir}")

    drive_service = setup_google_drive_service() if args.checkpoint_only_last else None

    model = Zi2ZiModel(
        input_nc=args.input_nc,
        embedding_num=args.embedding_num,
        embedding_dim=args.embedding_dim,
        ngf=args.ngf,
        ndf=args.ndf,
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        L1_penalty=args.L1_penalty,

        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        image_size=args.image_size,
        self_attention=args.self_attention,
        d_spectral_norm=args.d_spectral_norm,
        norm_type=args.norm_type,
        accum_steps=args.accum_steps,
        lr=args.lr,
        lr_D=args.lr_D
    )
    model.setup()
    model.print_networks(True)

    start_epoch = 0
    global_steps = 0
    if args.resume:
        print(f"Resumed model from step/epoch: {args.resume}")
        model_loaded = model.load_networks(args.resume)
        if not model_loaded:
            return

    train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), input_nc=args.input_nc)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total_batches = math.ceil(len(train_dataset) / args.batch_size)

    # --- Training Loop ---
    start_time = time.time()
    print(f"Starting training from epoch {start_epoch}/{args.epoch - 1}...")

    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()

        for batch_id, batch_data in enumerate(dataloader):
            labels, image_B, image_A = batch_data
            model_input_data = {'label': labels, 'A': image_A, 'B': image_B}
            model.set_input(model_input_data)
            losses = model.optimize_parameters()
            global_steps += 1

            if batch_id % 100 == 0:
                passed = time.time() - start_time
                log_format = "Epoch: [%2d], [%4d/%4d] time: %5d, d_loss: %.4f, g_loss: %.4f, " + \
                             "adv_loss: %.4f, const_loss: %.4f, l1_loss: %.4f, lambda_adv: %.2f"
                print(log_format % (epoch, batch_id, total_batches, passed, losses["d_loss"], model.g_loss.item(),
                                    losses["loss_adv"], losses["loss_const"], losses["loss_l1"], losses["lambda_adv"]))

            # --- Checkpointing ---
            if global_steps % args.checkpoint_steps == 0:
                if global_steps >= args.checkpoint_steps_after:
                    # You must save the data before deleting it; the worst-case scenario is that all data will be lost.
                    model.save_networks(global_steps)
                    # --- Clean up old checkpoints (Optional: only keep last) ---
                    if args.checkpoint_only_last:
                        # --- Clean up old checkpoints ---
                        for index_step in range(args.checkpoint_steps, global_steps, args.checkpoint_steps):
                            for net_type in ["G", "D"]:
                                f_path = f"{index_step}_net_{net_type}.pth"
                                try:
                                    os.remove(os.path.join(checkpoint_dir, f_path))
                                except FileNotFoundError:
                                    continue
                        clear_google_drive_trash(drive_service)
                else:
                    print(f"Checkpoint step {global_steps} reached, but saving starts after step {args.checkpoint_steps_after}.")

        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        print(f"--- End of Epoch {epoch} --- Time: {epoch_time:.0f}s ---")

        model.update_lr()

    # --- End of Training ---
    print("\n--- Training Finished ---")
    # Save the final model state
    print("Saving final model...")
    model.save_networks('latest') # Save with 'latest' label
    print("Final model saved.")

    total_training_time = time.time() - start_time
    print(f"Total Training Time: {total_training_time:.1f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='overwrite checkpoint dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--checkpoint_only_last', action='store_true', help='remove all previous versions, only keep last version')
    parser.add_argument('--checkpoint_steps', type=int, default=100, help='number of batches in between two checkpoints')
    parser.add_argument('--checkpoint_steps_after', type=int, default=1, help='save the number of batches after')
    parser.add_argument('--data_dir', type=str, default=None, help='overwrite data dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--image_size', type=int, default=256, help="size of your input and output image")
    parser.add_argument('--input_nc', type=int, default=1, help='number of input images channels')
    parser.add_argument('--L1_penalty', type=int, default=97, help='weight for L1 loss')
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')

    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--lr_D', type=float, default=None, help='initial learning rate for discriminator (default: same as lr)')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for random and pytorch')
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--sample_steps', type=int, default=10, help='number of batches in between two samples are drawn from validation set')
    parser.add_argument('--self_attention', action='store_true', help='use self attention in generator')
    parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
    parser.add_argument('--d_spectral_norm', action='store_true', help='use spectral normalization in discriminator')
    parser.add_argument('--norm_type', type=str, default="instance", help='normalization type: instance or batch')
    parser.add_argument('--accum_steps', type=int, default=1, help='梯度累積次數 (accumulate gradients for this many small batches)')

    main()
