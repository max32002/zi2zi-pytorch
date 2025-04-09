import argparse
import math
import os
import random
import time

import torch
#torch.autograd.set_detect_anomaly(True)  # 添加異常檢測
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

def train(args):
    """訓練主函數"""
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
        Lconst_penalty=args.Lconst_penalty,
        Lcategory_penalty=args.Lcategory_penalty,
        save_dir=checkpoint_dir,
        gpu_ids=args.gpu_ids,
        self_attention=args.self_attention,
        attention_type=args.attention_type,
        residual_block=args.residual_block,
        epoch=args.epoch,
        g_blur=args.g_blur,
        d_blur=args.d_blur,
        lr=args.lr,
        norm_type=args.norm_type
    )

    model.print_networks(True)

    start_epoch = 0
    global_steps = 0    
    if args.resume:
        print(f"Resumed model from step/epoch: {args.resume}")
        # If loading optimizer/scheduler state, you'd need to load those too and potentially update start_epoch/global_steps
        model_loaded = model.load_networks(args.resume)
        if not model_loaded:
            return

    train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), input_nc=args.input_nc)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total_batches = math.ceil(len(train_dataset) / args.batch_size)


    # --- Training Loop ---
    start_time = time.time()
    print(f"Starting training from epoch {start_epoch}...")

    for epoch in range(start_epoch, args.epoch):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch}/{args.epoch - 1} ---")

        for batch_id, batch_data in enumerate(dataloader):
            current_step_time = time.time()

            labels, image_B, image_A = batch_data
            model_input_data = {'label': labels, 'A': image_A, 'B': image_B}
            model.set_input(model_input_data)

            losses = model.optimize_parameters(args.use_autocast)
            global_steps += 1

            if batch_id % 100 == 0:
                elapsed_batch_time = time.time() - current_step_time
                total_elapsed_time_seconds = time.time() - start_time
                total_elapsed_hours = int(total_elapsed_time_seconds // 3600)
                total_elapsed_minutes = int((total_elapsed_time_seconds % 3600) // 60)
                total_elapsed_seconds = int(total_elapsed_time_seconds % 60)

                time_str = ""
                if total_elapsed_hours > 0:
                    time_str += f"{total_elapsed_hours}h "
                if total_elapsed_minutes > 0:
                    time_str += f"{total_elapsed_minutes}m "
                time_str += f"{total_elapsed_seconds}s"

                print(
                    f"Epoch: [{epoch:2d}], Batch: [{batch_id:4d}/{total_batches:4d}] "
                    f" | Time/Batch: {elapsed_batch_time:.2f}s | Total Time: {time_str}\n"
                    f" d_loss: {losses['d_loss']:.4f}, g_loss:  {losses['g_loss']:.4f}, "
                    f"const_loss: {losses['const_loss']:.4f}, l1_loss: {losses['l1_loss']:.4f}, fm_loss: {losses['fm_loss']:.4f}, perc_loss: {losses['perceptual_loss']:.4f}"
                )

            # --- Checkpointing ---
            if global_steps % args.checkpoint_steps == 0:
                if global_steps >= args.checkpoint_steps_after:
                    model.save_networks(global_steps)

                    # --- Clean up old checkpoints (Optional: only keep last) ---
                    if args.checkpoint_only_last:
                        for checkpoint_index in range(0, global_steps, args.checkpoint_steps):
                            for net_type in ["D", "G"]:
                                filepath = os.path.join(checkpoint_dir, f"{checkpoint_index}_net_{net_type}.pth")
                                if os.path.isfile(filepath):
                                    os.remove(filepath)
                        clear_google_drive_trash(drive_service)
                else:
                    print(f"\nCheckpoint step {global_steps} reached, but saving starts after step {args.checkpoint_steps_after}.")


        # --- End of Epoch ---
        epoch_time = time.time() - epoch_start_time
        print(f"\n--- End of Epoch {epoch} --- Time: {epoch_time:.2f}s ---")

        # Update Learning Rate Schedulers
        model.scheduler_G.step()
        model.scheduler_D.step()
        print(f"LR Scheduler stepped. Current LR G: {model.scheduler_G.get_last_lr()[0]:.6f}, LR D: {model.scheduler_D.get_last_lr()[0]:.6f}")


    # --- End of Training ---
    print("\n--- Training Finished ---")
    # Save the final model state
    print("Saving final model...")
    model.save_networks('latest') # Save with 'latest' label
    print("Final model saved.")

    total_training_time = time.time() - start_time
    print(f"Total Training Time: {total_training_time:.2f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--attention_type', type=str, default="linear", help="切換 Attention 的類型")
    parser.add_argument('--batch_size', type=int, default=16, help='number of examples in batch')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='overwrite checkpoint dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--checkpoint_only_last', action='store_true', help='remove all previous versions, only keep last version')
    parser.add_argument('--checkpoint_steps', type=int, default=100, help='number of batches in between two checkpoints')
    parser.add_argument('--checkpoint_steps_after', type=int, default=1, help='save the number of batches after')
    parser.add_argument('--d_blur', action='store_true')
    parser.add_argument('--data_dir', type=str, default=None, help='overwrite data dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
    parser.add_argument('--embedding_num', type=int, default=2, help="number for distinct embeddings")
    parser.add_argument('--epoch', type=int, default=100, help='number of epoch')
    parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--g_blur', action='store_true')
    parser.add_argument('--gpu_ids', default=[], nargs='+', help="GPUs")
    parser.add_argument('--input_nc', type=int, default=3, help='number of input images channels')
    parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
    parser.add_argument('--Lcategory_penalty', type=float, default=1.0, help='weight for category loss')
    parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
    parser.add_argument('--norm_type', type=str, default="instance", help='normalization type: instance or batch')
    parser.add_argument('--random_seed', type=int, default=777, help='random seed for random and pytorch')
    parser.add_argument('--residual_block', action='store_true')
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--use_autocast', action="store_true", help='Enable autocast for mixed precision training')
    args = parser.parse_args()
    train(args)
