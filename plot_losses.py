import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_log_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        log_text = file.read()

    pattern = re.compile(
        r"Epoch: \[\s*(\d+)\], Batch: \[\s*\d+/\s*\d+\].*?\n"
        r"\s*d_loss:\s*([\d.]+),\s*g_loss:\s*([\d.]+),\s*const_loss:\s*([\d.]+),\s*"
        r"l1_loss:\s*([\d.]+),\s*fm_loss:\s*([\d.]+),\s*perc_loss:\s*([\d.]+),\s*edge:\s*([\d.]+)"
    )

    records = []
    for match in pattern.finditer(log_text):
        epoch = int(match.group(1))
        d_loss = float(match.group(2))
        g_loss = float(match.group(3))
        const_loss = float(match.group(4))
        l1_loss = float(match.group(5))
        fm_loss = float(match.group(6))
        perc_loss = float(match.group(7))
        edge = float(match.group(8))

        records.append({
            'epoch': epoch,
            'd_loss': d_loss,
            'g_loss': g_loss,
            'const_loss': const_loss,
            'l1_loss': l1_loss,
            'fm_loss': fm_loss,
            'perc_loss': perc_loss,
            'edge': edge
        })

    return pd.DataFrame(records)

def plot_losses(df, output_filename=None):
    df_mean = df.groupby('epoch').mean().reset_index()

    plt.figure(figsize=(12, 7))
    loss_types = ['d_loss', 'g_loss', 'l1_loss', 'perc_loss', 'edge']
    for loss in loss_types:
        plt.plot(df_mean['epoch'], df_mean[loss], label=loss)

    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename)
        print(f"圖檔已儲存至: {output_filename}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從 log 檔案中解析 loss 並繪製圖表。")
    parser.add_argument("log_file", help="輸入的 log 檔案名稱")
    parser.add_argument("-s", "--silent", action="store_true", help="不顯示 plot，直接儲存圖檔")
    args = parser.parse_args()

    log_file = args.log_file
    silent_mode = args.silent

    try:
        df = parse_log_file(log_file)
        base_filename = os.path.splitext(log_file)[0]
        output_filename = f"{base_filename}.png"
        plot_losses(df, output_filename if silent_mode else None)
    except FileNotFoundError:
        print(f"找不到檔案: {log_file}")
    except Exception as e:
        print(f"發生錯誤: {e}")