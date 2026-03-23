from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def compute_error_metrics(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'mae': float(mae),
        'mape': float(mape),
        'mse': float(mse),
        'rmse': float(rmse),
    }


def save_metrics_json(metrics_payload, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open('w', encoding='utf-8') as file:
        json.dump(metrics_payload, file, indent=2)


def plot_static_prediction(curr, volt, pred, date, lower=None, upper=None):
    t = np.arange(len(volt))
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(t, curr, color='g', label='Current')
    ax1.set_ylabel('Current (A)', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_xticks([0])
    ax1.set_xticklabels([date], rotation=45, ha='right')

    ax2.plot(t, volt, '-r', label='Voltage (true)')
    if lower is not None and upper is not None:
        ax2.plot(t, pred, '-b', label='Voltage (mean pred)')
        ax2.fill_between(t, lower, upper, color='b', alpha=0.3, label='95% CI')
    else:
        ax2.plot(t, pred, '--r', label='Voltage (pred)')
    ax2.set_ylabel('Voltage (V)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc='upper right')
    return fig, ax1, ax2


def plot_rolling_prediction(volt, pred, curr=None, lower=None, upper=None, title=None, show_current_on_secondary_axis=False):
    t = np.arange(len(volt))
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(t, volt, '--k', label='True Voltage' if curr is not None else 'True')
    ax1.plot(t, pred, 'b-', label='Pred Mean Voltage' if curr is not None else 'Mean pred')
    if lower is not None and upper is not None:
        label = '95% CI Voltage' if curr is not None else '95% CI'
        ax1.fill_between(t, lower, upper, alpha=0.3, color='b', label=label)
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Voltage (V)')

    if title:
        ax1.set_title(title)

    axes = [ax1]
    if curr is not None:
        if show_current_on_secondary_axis:
            ax2 = ax1.twinx()
            ax2.plot(t, curr, '-r', label='Current')
            ax2.set_ylabel('Current (A)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            axes.append(ax2)
        else:
            ax1.plot(t, curr, 'g-', label='Current')

    lines = []
    labels = []
    for axis in axes:
        axis_lines, axis_labels = axis.get_legend_handles_labels()
        lines.extend(axis_lines)
        labels.extend(axis_labels)
    ax1.legend(lines, labels, loc='upper right')
    return fig, axes
