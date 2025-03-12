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
    # else:
    #     print("drive_service is None")


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
        image_size=args.image_size,
        self_attention=args.self_attention,
        residual_block=args.residual_block,
        final_channels=args.final_channels,
        epoch=args.epoch,
        g_blur=args.g_blur,
        d_blur=args.d_blur,
        lr=args.lr,
    )

    model.setup()
    model.print_networks(True)
    if args.resume:
        model.load_networks(args.resume)

    train_dataset = DatasetFromObj(os.path.join(data_dir, 'train.obj'), input_nc=args.input_nc)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    total_batches = math.ceil(len(train_dataset) / args.batch_size)

    global_steps = 0
    start_time = time.time()  # 確保 start_time 被賦值
    for epoch in range(args.epoch):
        for batch_id, batch in enumerate(dataloader):
            model.set_input(batch[0], batch[2], batch[1])
            const_loss, l1_loss, cheat_loss, fm_loss, vgg_loss = model.optimize_parameters(args.use_autocast)

            if batch_id % 100 == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch: [{epoch:2d}], [{batch_id:4d}/{total_batches:4d}] "
                    f"time: {elapsed_time:.0f}, d_loss: {model.d_loss.item():.5f}, "
                    f"g_loss: {model.g_loss.item():.5f}, cheat_loss: {cheat_loss:.5f}, "
                    f"const_loss: {const_loss:.5f}, l1_loss: {l1_loss:.5f}, fm_loss: {fm_loss:.5f}, vgg_loss: {vgg_loss:.5f}"
                )

            if global_steps % args.checkpoint_steps == 0:
                if global_steps >= args.checkpoint_steps_after:
                    print(f"checkpoint: current step {global_steps}")
                    model.save_networks(global_steps)
                    if args.checkpoint_only_last:
                        for checkpoint_index in range(0, global_steps, args.checkpoint_steps):
                            for net_type in ["D", "G"]:
                                filepath = os.path.join(checkpoint_dir, f"{checkpoint_index}_net_{net_type}.pth")
                                if os.path.isfile(filepath):
                                    os.remove(filepath)
                        clear_google_drive_trash(drive_service)
                else:
                    print(f"checkpoint: current step {global_steps}，save after step {args.checkpoint_steps_after}")
            global_steps += 1

        if (epoch + 1) % args.schedule == 0:
            model.update_lr()

    model.save_networks(global_steps)
    elapsed_time = time.time() - start_time
    print(f"經過時間：{elapsed_time:4d} 秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
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
    parser.add_argument('--embedding_num', type=int, default=40, help="number for distinct embeddings")
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
    parser.add_argument('--final_channels', type=int, default=1, help="discriminator final channels")
    parser.add_argument('--g_blur', action='store_true')
    parser.add_argument('--d_blur', action='store_true')
    parser.add_argument("--use_autocast", action="store_true", help="Enable autocast for mixed precision training")
    args = parser.parse_args()
    train(args)