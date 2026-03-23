from __future__ import annotations

from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import BatteryRunDataset
from .evaluation import compute_error_metrics, plot_rolling_prediction
from .predict import predict_with_uncertainty
from .trainer import fine_tune_battery_model, load_battery_checkpoint, save_battery_checkpoint


def _build_model(model_cls, args):
    if args.model_name != 'BattNN':
        raise ValueError(f"Unsupported model: {args.model_name}")
    model = model_cls(args)
    print('Selected model: BattNN')
    return model


def _fit_model(model_cls, args, train_x, train_y):
    model = _build_model(model_cls, args)
    model, _ = fine_tune_battery_model(model, train_x, train_y, args)
    return model


def _run_rolling_experiment(
    model_cls,
    args,
    block_size: int,
    mc_samples: int,
    save_dir: str,
    show_current_on_secondary_axis: bool,
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    dataset = BatteryRunDataset.from_directory(args.npz_dir)
    if dataset.removed_runs:
        print('Removed neg-current runs:', list(dataset.removed_runs))

    files = dataset.files
    run_by_name = {run.file_name: run for run in dataset.runs}
    total_runs = len(files)
    print(f'{total_runs} total runs, block size = {block_size}')

    init_files = files[:block_size]
    train_x, train_y, _ = dataset.rolling_block(init_files).load_arrays(args.seq_len)
    init_args = copy(args)
    init_args.batch_size = block_size
    init_args.train_runs = block_size
    model = _fit_model(model_cls, init_args, train_x, train_y)

    iteration_metrics = []
    per_run_metrics = []

    for iteration, start in enumerate(range(block_size, total_runs, block_size), start=1):
        test_files = files[start:start + block_size]
        print(f'\nIteration {iteration}: predicting runs {start}–{start + len(test_files) - 1}')

        block_metrics = []
        dates_block = []

        for run_index, file_name in enumerate(test_files, start=1):
            run = run_by_name[file_name]
            curr = run.current
            volt = run.voltage
            date = run.date
            dates_block.append(date)

            input_tensor = torch.from_numpy(curr).to(args.device).view(1, -1)
            prediction = predict_with_uncertainty(model, input_tensor, mc_samples=mc_samples, n_jobs=1)
            mean_pred = prediction['mean']
            lower = prediction['lower']
            upper = prediction['upper']

            metrics_payload = compute_error_metrics(volt, mean_pred)
            metric_vector = [
                metrics_payload['mae'],
                metrics_payload['mape'],
                metrics_payload['mse'],
                metrics_payload['rmse'],
            ]
            block_metrics.append(metric_vector)

            fig, axes = plot_rolling_prediction(
                volt,
                mean_pred,
                curr=curr if show_current_on_secondary_axis else None,
                lower=lower,
                upper=upper,
                title=f'Iter {iteration} Run {run_index}: {file_name}',
                show_current_on_secondary_axis=show_current_on_secondary_axis,
            )
            axes[0].text(
                0.5,
                0.95,
                f"MSE = {metrics_payload['mse']:.4e}",
                transform=axes[0].transAxes,
                ha='center',
                va='top',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7),
            )
            fig.tight_layout()
            fig.show()

        block_metrics = np.array(block_metrics)
        per_run_metrics.append(block_metrics)
        mean_metrics = block_metrics.mean(axis=0)
        print(f'Iteration {iteration} summary:')
        for name, value in zip(['MAE', 'MAPE', 'MSE', 'RMSE'], mean_metrics):
            print(f'  {name} = {value:.4e}')
        iteration_metrics.append(mean_metrics)

        last_date = dates_block[-1]
        safe_date = last_date.replace(' ', '_').replace(':', '-')
        checkpoint_path = Path(save_dir) / f"{args.model_name}_{safe_date}.pth"
        save_battery_checkpoint(model, save_path=checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')

        next_args = copy(args)
        next_args.batch_size = len(test_files)
        next_args.train_runs = len(test_files)
        next_model = _build_model(model_cls, init_args)
        load_battery_checkpoint(next_model, checkpoint_path, map_location=args.device)
        next_model = next_model.to(args.device)

        fine_tune_x, fine_tune_y, _ = dataset.rolling_block(test_files).load_arrays(args.seq_len)
        next_model.config = next_args
        next_model.set_batch_size(next_args.batch_size)
        model, _ = fine_tune_battery_model(next_model, fine_tune_x, fine_tune_y, next_args)

    iteration_metrics = np.stack(iteration_metrics)
    iterations = np.arange(1, iteration_metrics.shape[0] + 1)
    plt.figure(figsize=(8, 4))
    for index, name in enumerate(['MAE', 'MAPE', 'MSE', 'RMSE']):
        plt.plot(iterations, iteration_metrics[:, index], '-o', label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Metrics per Rolling Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return iteration_metrics, per_run_metrics


def run_deterministic_rolling_experiment(model_cls, args, block_size: int, mc_samples: int, save_dir: str):
    return _run_rolling_experiment(
        model_cls=model_cls,
        args=args,
        block_size=block_size,
        mc_samples=mc_samples,
        save_dir=save_dir,
        show_current_on_secondary_axis=False,
    )


def run_variational_rolling_experiment(model_cls, args, block_size: int, mc_samples: int, save_dir: str):
    return _run_rolling_experiment(
        model_cls=model_cls,
        args=args,
        block_size=block_size,
        mc_samples=mc_samples,
        save_dir=save_dir,
        show_current_on_secondary_axis=True,
    )
