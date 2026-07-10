import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

'''
the purpose of this function is to generate plots of training and validation loss per epoch
'''

def main(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    loss_cols_train = [c for c in df.columns if "train" in c and "loss" in c]
    loss_cols_val = [c for c in df.columns if "val" in c and "loss" in c]

    plt.figure()
    for col in loss_cols_train:
        plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training losses — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "train_losses.png")
    plt.figure()
    for col in loss_cols_val:
        plt.plot(df["epoch"], df[col], label=col)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Validation losses — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "val_losses.png")

    print(f"saved -> {run_dir}/train_losses.png, val_losses.png")
    print("\nColumns found in results.csv:")
    print(list(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="e.g. ~/runs/cme_yolo/yolo26s_50pct")
    args = parser.parse_args()
    main(args)
