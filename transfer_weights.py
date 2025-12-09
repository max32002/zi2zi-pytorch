import argparse
import os

import torch
import torch.nn as nn

from model.model import UNetGenerator


def get_layer_level(key):
    return len(key.split('.'))

def transfer_weights(old_state_dict, new_model, show_success=True):
    new_state = new_model.state_dict()
    transfer_count = 0
    copied_keys = set()
    old_keys_not_in_new = set(old_state_dict.keys()) - set(new_state.keys())
    suggested_mapping = {}
    shape_mismatch_key = []
    missing_key = []

    print("--- Direct Copy Attempt ---", len(new_state.items()))
    for new_key, new_param in new_state.items():
        if new_key in old_state_dict:
            old_param = old_state_dict[new_key]
            if new_param.shape == old_param.shape:
                new_param.copy_(old_param)
                if show_success:
                    print(f"✅ Copied (direct): {new_key}")
                transfer_count += 1
                copied_keys.add(new_key)
            else:
                print(f"⚠️ Shape mismatch (direct): {new_key} (New Shape: {tuple(new_param.shape)}, Old Shape: {tuple(old_param.shape)})")
                shape_mismatch_key.append(new_key)
        else:
            print(f"❌ Missing key (direct): {new_key} in old checkpoint")
            missing_key.append(new_key)

    print("-" * 30)
    print("--- Layers in Old Checkpoint but not in New Model ---")
    for old_key in sorted(list(old_keys_not_in_new)):
        print(f"⚠️ Extra key in old checkpoint: {old_key} (Shape: {tuple(old_state_dict[old_key].shape)}, Level: {get_layer_level(old_key)})")
        suggested_matches = []
        old_param_shape = old_state_dict[old_key].shape
        old_layer_level = get_layer_level(old_key)
        for new_k, new_p in new_state.items():
            if new_k not in copied_keys and new_p.shape == old_param_shape and get_layer_level(new_k) == old_layer_level:
                suggested_matches.append(new_k)
        if suggested_matches:
            print(f"   💡 Possible matches in new model (same shape & level): {suggested_matches}")
            if len(suggested_matches) == 1:
                suggested_mapping[suggested_matches[0]] = old_key

    print("-" * 30)
    print("--- Changed Mapping Attempt ---")
    changed_mapping = {
    }

    # 接著處理 changed_mapping 中的 layer
    for new_key, old_key in changed_mapping.items():
        if new_key not in copied_keys and new_key in new_state:
            if old_key in old_state_dict:
                if old_state_dict[old_key].shape == new_state[new_key].shape:
                    new_state[new_key].copy_(old_state_dict[old_key])
                    print(f"✅ Copied (mapped): {old_key} → {new_key}")
                    transfer_count += 1
                else:
                    print(f"⚠️ Shape mismatch (mapped): {old_key} vs {new_key} (Old Shape: {tuple(old_state_dict[old_key].shape)}, New Shape: {tuple(new_state[new_key].shape)})")
            else:
                print(f"❌ Missing key (mapped): {old_key} in old checkpoint")
        elif new_key in copied_keys:
            print(f"⏭️ Skipped (already copied): {new_key}")
        elif new_key not in new_state:
            print(f"❌ Missing key (new model): {new_key}")

    print("-" * 30)
    print(f"✅ Total transferred (direct + manual): {transfer_count} ({len(copied_keys)} + {transfer_count-len(copied_keys)}) layers")
    print(f"✅ Total key (new model): {len(new_state.items())} layers")
    print(f"⚠️ Total Shape mismatch key (new model): {len(shape_mismatch_key)} layers")
    print(f"❌ Total Missing key (new model): {len(missing_key)} layers")
    print("-" * 30)
    print("💡 Suggested mapping (excluding already mapped old keys):")
    filtered_suggested_mapping = {
        new_key: old_key
        for new_key, old_key in suggested_mapping.items()
        if old_key not in changed_mapping.values()
    }
    print(filtered_suggested_mapping)
    print("-" * 30)
    print("❌ Missing key:")
    filtered_missing_key = [
        new_key
        for new_key in missing_key
        if new_key not in changed_mapping.keys()
    ]
    print(filtered_missing_key)
    print("-" * 30)

    new_model.load_state_dict(new_state)
    return new_model

def transfer(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 原始模型的權重
    step = args.resume
    old_filepath_G = os.path.join(args.checkpoint_dir, f"{step}_net_G.pth")
    old_ckpt = torch.load(old_filepath_G, map_location="cpu")

    #input_nc = 1
    ngf = 64
    #self_attention = True
    #attention_type = "self"
    #up_mode = "pixelshuffle"
    #up_mode = "upsample"
    #embedding_dim = 64
    #embedding_num = 2
    norm_layer = nn.InstanceNorm2d
    model = UNetGenerator(
            input_nc=args.input_nc,
            output_nc=args.input_nc,
            ngf=ngf,
            embedding_num=args.embedding_num,
            embedding_dim=args.embedding_dim,
            self_attention=args.self_attention,
            attention_type=args.attention_type,
            attn_layers=[4, 6],
            norm_layer=norm_layer,
            up_mode=args.up_mode)

    show_success = False
    new_model = transfer_weights(old_ckpt, model, show_success=show_success)

    # 保存新的模型權重
    new_filepath_G = os.path.join(args.checkpoint_dir, f"new_net_G.pth")
    torch.save(new_model.state_dict(), new_filepath_G)
    print(f"✅ New model weights saved to: {new_filepath_G}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer')
    parser.add_argument('--input_nc', type=int, default=1, help='number of input images channels')
    parser.add_argument('--attention_type', type=str, default="linear", help="切換 Attention 的類型")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='overwrite checkpoint dir path, if data dir is not same with checkpoint dir')
    parser.add_argument('--embedding_dim', type=int, default=64, help="dimension for embedding")
    parser.add_argument('--embedding_num', type=int, default=2, help="number for distinct embeddings")
    parser.add_argument('--experiment_dir', required=True, help='experiment directory, data, samples,checkpoints,etc')
    parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
    parser.add_argument('--self_attention', action='store_true')
    parser.add_argument('--up_mode', type=str, default="conv", help="切換 upsample / conv / pixelshuffle")
    args = parser.parse_args()
    transfer(args)
