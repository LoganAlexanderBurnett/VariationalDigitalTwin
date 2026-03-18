import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm

METRICS = ['r2', 'mae', 'mape', 'rmse', 'rmspe']
X_END = 340070
SEQ_LEN = 5485


def get_shrink_dirs(base_dir: Path):
    return sorted(
        [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith('shrink_train_')],
        key=lambda path: int(path.name.split('_')[2]),
    )


def load_metrics(base_dir: Path):
    records = []
    for directory in get_shrink_dirs(base_dir):
        metrics_path = directory / 'performance_metrics.json'
        if not metrics_path.exists():
            print(f'Warning: no metrics in {directory.name}, skipping')
            continue

        with metrics_path.open() as file:
            metrics = json.load(file)

        n_removed = int(directory.name.split('_')[2])
        record = {'N_removed': n_removed}
        for output in ['output_1', 'output_2']:
            for metric in METRICS:
                record[f'{output}_{metric}'] = metrics.get(output, {}).get(metric)
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError('No metrics found in shrink_train directories.')

    df = df.sort_values('N_removed').reset_index(drop=True)
    df['N_sensors'] = df['N_removed'].iloc[::-1].values
    return df


def plot_metrics(df: pd.DataFrame):
    for metric in METRICS:
        plt.figure()
        plt.plot(df['N_sensors'], df[f'output_1_{metric}'], '--o', color='r', label='TS', markersize=3)
        plt.plot(df['N_sensors'], df[f'output_2_{metric}'], '--o', color='b', label='TF', markersize=3)
        plt.xlabel('Number of Training Sensors (N)')
        ylabel = 'R²' if metric == 'r2' else metric.upper()
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs N')
        plt.legend()
        plt.show()


def load_rescaled_runs(base_dir: Path):
    runs = []
    for directory in get_shrink_dirs(base_dir):
        n_removed = int(directory.name.split('_')[2])
        true_file = directory / 'Ytrue_rescaled.npy'
        pred_file = directory / 'Ypred_rescaled.npy'
        if true_file.exists() and pred_file.exists():
            runs.append((n_removed, np.load(true_file), np.load(pred_file)))
    return runs


def plot_predictions(runs, sample_step: int = 5):
    sampled = runs[::sample_step]
    if not sampled:
        raise RuntimeError('No valid runs found to plot.')

    cmap = cm.get_cmap('coolwarm', len(sampled))
    for out_idx in [0, 1]:
        plt.figure(figsize=(16, 6))
        for i, (n_removed, y_true, y_pred) in enumerate(sampled):
            length = len(y_pred)
            offset = X_END - length
            x_vals = np.arange(offset, offset + length)

            if i == 0:
                plt.plot(x_vals, y_true[:, out_idx], color='black', linewidth=2, label='True')
            else:
                plt.plot(x_vals, y_true[:, out_idx], color='black', linewidth=1, alpha=0.3)

            plt.plot(x_vals, y_pred[:, out_idx], '--', color=cmap(i), alpha=1, label=f'Pred N={n_removed} removed')

        plt.xlabel('Time Step')
        ylabel = 'TS' if out_idx == 0 else 'TF'
        plt.ylabel('Temperature (°C)')
        plt.title(f'True vs Predicted for {ylabel} (every {sample_step}th N, aligned end)')
        plt.xlim(X_END - 20 * SEQ_LEN, X_END)
        plt.ylim(0, 1200)
        plt.legend(ncol=3)

        for x_val in np.arange(SEQ_LEN, X_END, SEQ_LEN):
            plt.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

        plt.tight_layout()
        plt.show()


def plot_errors(runs, sample_step: int = 10):
    sampled = runs[::sample_step]
    if not sampled:
        raise RuntimeError('No valid runs found to plot.')

    cmap = cm.get_cmap('coolwarm', len(sampled))
    for out_idx in [0, 1]:
        plt.figure(figsize=(16, 6))
        for i, (n_removed, y_true, y_pred) in enumerate(sampled):
            length = len(y_pred)
            offset = X_END - length
            x_vals = np.arange(offset, offset + length)
            error = np.abs(y_true[:, out_idx] - y_pred[:, out_idx])
            plt.plot(x_vals, error, color=cmap(i), alpha=0.8, label=f'N={n_removed}')

        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        ylabel = 'Error in TS' if out_idx == 0 else 'Error in TF'
        plt.xlabel('Time Step')
        plt.ylabel('True − Predicted')
        plt.title(f'Prediction Error for {ylabel} (every {sample_step}th N), aligned ends')
        plt.legend(ncol=3)
        plt.xlim(X_END - 20 * SEQ_LEN, X_END)
        plt.ylim(0, 400)
        plt.tight_layout()
        plt.show()


def main():
    base_dir = Path(__file__).resolve().parent
    df = load_metrics(base_dir)
    print(df)
    plot_metrics(df)
    runs = load_rescaled_runs(base_dir)
    plot_predictions(runs)
    plot_errors(runs)


if __name__ == '__main__':
    main()
