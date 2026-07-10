import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Fallback loss gains if args.yaml doesn't specify them — these are general
# Ultralytics YOLOv8-era defaults, NOT verified against YOLO26 specifically.
# Always prefer whatever args.yaml actually contains for a given run.
DEFAULT_GAINS = {
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "seg": 7.5,
}


def get_gains(run_dir):
    gains = dict(DEFAULT_GAINS)
    args_path = run_dir / "args.yaml"
    if args_path.exists():
        with open(args_path) as f:
            saved = yaml.safe_load(f)
        for key in gains:
            if key in saved:
                gains[key] = saved[key]
    return gains


def weighted_total(df, cols, gains):
    total = None
    for col in cols:
        key = col.split("/")[-1].replace("_loss", "")
        gain = gains.get(key)
        if gain is None:
            print(f"  WARNING: no gain for '{key}' ({col}) — excluded from total_loss")
            continue
        term = df[col] * gain
        total = term if total is None else total + term
    return total


def main(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    results_csv = run_dir / "results.csv"

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    loss_cols_train = [c for c in df.columns if "train" in c and "loss" in c]
    loss_cols_val = [c for c in df.columns if "val" in c and "loss" in c]

    gains = get_gains(run_dir)

    # --- train losses ---
    plt.figure()
    for col in loss_cols_train:
        plt.plot(df["epoch"], df[col], label=col)

    train_total = weighted_total(df, loss_cols_train, gains)
    if train_total is not None:
        plt.plot(df["epoch"], train_total, label="total_loss", color="black", linewidth=2)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Training losses — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "train_losses.png")

    # --- val losses ---
    plt.figure()
    for col in loss_cols_val:
        plt.plot(df["epoch"], df[col], label=col)

    val_total = weighted_total(df, loss_cols_val, gains)
    if val_total is not None:
        plt.plot(df["epoch"], val_total, label="total_loss", color="black", linewidth=2)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Validation losses — {run_dir.name}")
    plt.legend()
    plt.savefig(run_dir / "val_losses.png")

    print(f"saved -> {run_dir}/train_losses.png, val_losses.png")
    print("\nColumns found in results.csv:")
    print(list(df.columns))
    print("\nGains used:", gains)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="e.g. ~/runs/cme_yolo/yolo26n_30pct_20ep")
    args = parser.parse_args()
    main(args)
