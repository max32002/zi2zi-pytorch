import re
import sys
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_losses(df):
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
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python plot_losses.py <log檔案名稱>")
        sys.exit(1)

    log_file = sys.argv[1]
    try:
        df = parse_log_file(log_file)
        plot_losses(df)
    except FileNotFoundError:
        print(f"找不到檔案: {log_file}")
    except Exception as e:
        print(f"發生錯誤: {e}")
