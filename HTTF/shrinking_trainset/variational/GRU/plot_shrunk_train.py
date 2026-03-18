from pathlib import Path

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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


def load_probabilistic_runs(base_dir: Path):
    runs = []
    for directory in get_shrink_dirs(base_dir):
        n_removed = int(directory.name.split('_')[2])
        true_file = directory / 'Ytrue.npy'
        pred_file = directory / 'Ypred.npy'
        if true_file.exists() and pred_file.exists():
            runs.append((n_removed, np.load(true_file), np.load(pred_file)))
    return runs


def plot_prediction_intervals(runs, sample_step: int = 10, output_path: str = 'HTTF_remove_sensors_preds.png'):
    sampled = runs[::sample_step]
    if not sampled:
        raise RuntimeError('No valid runs found to plot.')

    all_n = [entry[0] for entry in sampled]
    norm = Normalize(vmin=min(all_n), vmax=max(all_n))
    cmap = cm.get_cmap('coolwarm')
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    fig = plt.figure(figsize=(16, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[0, 2])

    for out_idx, axis in enumerate(axes):
        for i, (n_removed, y_true, y_pred) in enumerate(sampled):
            length = y_pred.shape[0]
            offset = X_END - length
            x_vals = np.arange(offset, offset + length)

            if i == 0:
                axis.plot(x_vals, y_true[:, out_idx], color='black', linewidth=2, label='True')
            else:
                axis.plot(x_vals, y_true[:, out_idx], color='black', linewidth=1, alpha=0.3)

            mean_pred = y_pred[:, out_idx, 0]
            lower_ci = y_pred[:, out_idx, 1]
            upper_ci = y_pred[:, out_idx, 2]
            color = scalar_mappable.to_rgba(n_removed)

            axis.plot(x_vals, mean_pred, linestyle='--', color=color, alpha=1.0)
            axis.fill_between(x_vals, lower_ci, upper_ci, color=color, alpha=0.1)

        if out_idx == 0:
            axis.set_ylabel('Solid Temperature (°C)', fontsize=12)
            axis.legend(loc='upper left')
        else:
            axis.set_ylabel('Fluid Temperature (°C)', fontsize=12)

        axis.set_xlabel('Time Step', fontsize=12)
        axis.set_xlim(X_END - 20 * SEQ_LEN, X_END)
        axis.set_ylim(0, 1200)

        for x_val in np.arange(SEQ_LEN, X_END, SEQ_LEN):
            axis.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    colorbar = fig.colorbar(scalar_mappable, cax=color_axis, orientation='vertical')
    colorbar.set_label('Sensors removed from training', fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_interval_widths(runs, sample_step: int = 10):
    sampled = runs[::sample_step]
    if not sampled:
        raise RuntimeError('No valid runs found to plot.')

    all_n = [entry[0] for entry in sampled]
    norm = Normalize(vmin=min(all_n), vmax=max(all_n))
    cmap = cm.get_cmap('coolwarm')
    scalar_mappable = ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    fig = plt.figure(figsize=(16, 6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    color_axis = fig.add_subplot(grid[0, 2])

    for out_idx, axis in enumerate(axes):
        for n_removed, _y_true, y_pred in sampled:
            length = y_pred.shape[0]
            offset = X_END - length
            x_vals = np.arange(offset, offset + length)
            interval_width = y_pred[:, out_idx, 2] - y_pred[:, out_idx, 1]
            axis.plot(x_vals, interval_width, linestyle='--', color=scalar_mappable.to_rgba(n_removed), alpha=1.0)

        axis.set_xlabel('Time Step', fontsize=12)
        if out_idx == 0:
            axis.set_ylabel('Solid Temperature (°C)', fontsize=12)
        else:
            axis.set_ylabel('Fluid Temperature (°C)', fontsize=12)
        axis.set_xlim(X_END - 20 * SEQ_LEN, X_END)
        axis.set_ylim(0, 1200)

        for x_val in np.arange(SEQ_LEN, X_END, SEQ_LEN):
            axis.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    colorbar = fig.colorbar(scalar_mappable, cax=color_axis, orientation='vertical')
    colorbar.set_label('Sensors removed from training', fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    base_dir = Path(__file__).resolve().parent
    df = load_metrics(base_dir)
    print(df)
    plot_metrics(df)
    runs = load_probabilistic_runs(base_dir)
    plot_prediction_intervals(runs)
    plot_interval_widths(runs)


if __name__ == '__main__':
    main()
