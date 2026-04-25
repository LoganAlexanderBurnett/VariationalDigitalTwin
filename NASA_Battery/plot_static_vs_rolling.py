from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FIGURE = BASE_DIR / "BattNNStaticVsRolling.png"
OUTPUT_CSV = BASE_DIR / "BattNNStaticVsRolling.csv"


def load_metrics_from_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    key = data.files[0]
    arr = data[key]  # [MAE, MAPE, MSE, RMSE]
    mape = arr[:, 1]
    mse = arr[:, 2]
    rmse = arr[:, 3]
    return mse, mape, rmse


def plot_aligned_runs(npz_dirs, label_map=None, color_map=None) -> None:
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    default_idx = 0

    metrics = []
    for dirpath in npz_dirs:
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith(".npz"):
                continue
            full_path = os.path.join(dirpath, fname)
            mse, mape, _ = load_metrics_from_npz(full_path)
            metrics.append((fname, len(mse), mse, mape))

    if not metrics:
        raise RuntimeError(f"No .npz files found in {npz_dirs}")

    max_n = max(item[1] for item in metrics)
    aligned_data = []
    for fname, n_steps, mse, mape in metrics:
        x_values = np.arange(max_n - n_steps + 1, max_n + 1)
        aligned_data.append((fname, x_values, mse, mape))

    fig, (ax_mse, ax_mape) = plt.subplots(ncols=2, figsize=(14, 5), sharex=True)
    for fname, x_values, mse, mape in aligned_data:
        label = label_map.get(fname, fname) if label_map else fname
        if color_map and fname in color_map:
            color = color_map[fname]
        else:
            color = default_colors[default_idx % len(default_colors)]
            default_idx += 1

        ax_mse.scatter(x_values, mse, color=color, alpha=0.3, label=label)
        ax_mape.scatter(x_values, mape, color=color, alpha=0.3, label=label)

    ax_mse.set_xlabel("Discharge # (Chronologically Ordered)")
    ax_mse.set_ylabel("MSE ($V^2$)")
    ax_mse.grid(alpha=0.3)
    ax_mse.legend(loc="best", fontsize=9)

    ax_mape.set_xlabel("Discharge # (Chronologically Ordered)")
    ax_mape.set_ylabel("MAPE (%)")
    ax_mape.grid(alpha=0.5)

    plt.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close(fig)


def collect_average_metrics(npz_dirs) -> pd.DataFrame:
    records = []
    for dirpath in npz_dirs:
        if not os.path.isdir(dirpath):
            continue
        for fname in os.listdir(dirpath):
            if not fname.endswith(".npz"):
                continue
            full_path = os.path.join(dirpath, fname)
            mse, mape, rmse = load_metrics_from_npz(full_path)
            records.append(
                {
                    "filename": fname,
                    "avg_mse": float(np.mean(mse)),
                    "avg_mape": float(np.mean(mape)),
                    "avg_rmse": float(np.mean(rmse)),
                }
            )

    if not records:
        raise RuntimeError(f"No .npz files found in {npz_dirs}")

    return pd.DataFrame(records)


def main() -> None:
    dirs = [
        BASE_DIR / "static_training" / "variational" / "results",
        BASE_DIR / "rolling_training" / "variational" / "results",
    ]
    labels = {
        "vBattNN-NASA-batch_size=30-seq_len=30.npz": "30 discharges",
        "vBattNN-NASA-batch_size=60-seq_len=30.npz": "60 discharges",
        "vBattNN-NASA-batch_size=150-seq_len=30.npz": "150 discharges",
        "vBattNN-NASA-batch_size=300-seq_len=30.npz": "300 discharges",
        "vBattNN-NASA-batch_size=500-seq_len=30.npz": "500 discharges",
        "RollingVBattNN-NASA-batch_size=30-seq_len=30.npz": "Digital Twin update every 10 discharges",
    }
    colors = {
        "vBattNN-NASA-batch_size=30-seq_len=30.npz": "deeppink",
        "vBattNN-NASA-batch_size=60-seq_len=30.npz": "dodgerblue",
        "vBattNN-NASA-batch_size=150-seq_len=30.npz": "gold",
        "vBattNN-NASA-batch_size=300-seq_len=30.npz": "deepskyblue",
        "vBattNN-NASA-batch_size=500-seq_len=30.npz": "crimson",
        "RollingVBattNN-NASA-batch_size=30-seq_len=30.npz": "gray",
    }

    plot_aligned_runs([str(path) for path in dirs], label_map=labels, color_map=colors)

    df_metrics = collect_average_metrics([str(path) for path in dirs])
    df_metrics.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {OUTPUT_FIGURE}")
    print(f"Saved {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
