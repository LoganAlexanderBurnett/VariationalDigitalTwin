import os
import copy
import random
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from vBattNN import BattNN

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import (
    BatteryRunDataset,
    compute_error_metrics,
    fine_tune_battery_model,
    load_battery_checkpoint,
    plot_rolling_prediction,
    predict_with_uncertainty,
    save_battery_checkpoint,
)

# reproducibility
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)


def train(args, train_x, train_y, model_name='BattNN'):
    if model_name == 'BattNN':
        model = BattNN(args)
        print("Selected model: BattNN")
    elif model_name == 'LSTM':
        model = LSTM(args)
        print("Selected model: LSTM")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model, _ = fine_tune_battery_model(model, train_x, train_y, args)
    return model
    

def get_args():
    parser = argparse.ArgumentParser(description='Battery Net for NPZData')
    # — Original battery‐model params —
    parser.add_argument('--V0',        type=float, default=4.2)
    parser.add_argument('--x0',        nargs=3,   type=float, default=[8000, 0, 0])
    parser.add_argument('--dt',        type=float, default=1.0)
    parser.add_argument('--VEOD',      type=float, default=3.2)
    parser.add_argument('--Rp',        type=float, default=1000)
    parser.add_argument('--Rs',        type=float, default=0.5)
    parser.add_argument('--Csp',       type=float, default=15)
    parser.add_argument('--Cs',        type=float, default=500)

    # — Training / data params —
    parser.add_argument('--train-runs', '--batch_size', '-n', dest='train_runs', type=int, default=30)
    parser.add_argument('--seq_len',    '-l', type=int,   default=30)
    parser.add_argument('--npz_dir',              default='../../dataset/')

    # — Optimization params —
    parser.add_argument('--device',     default='cpu')
    parser.add_argument('--epoch',      type=int,   default=1000)
    parser.add_argument('--lr',         type=float, default=2e-2)
    parser.add_argument('--weight_decay',type=float,default=5e-4)

    # — Model selection & saving —
    parser.add_argument('--model_name', choices=['BattNN','LSTM'], default='BattNN')
    parser.add_argument('--save_model', choices=[None,'NASA'], default='NASA')
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--save-dir', default='./models')
    parser.add_argument('--mc_samples', type=int, default=250)

    # ignore Jupyter args
    args, _ = parser.parse_known_args()
    return args

# ── Rolling UQ digital‐twin demo ───────────────────────────────────────────────
def rolling_fine_tune_uq(npz_dir, seq_len, block=5,
                         model_name='BattNN', args=None,
                         mc_samples=100,
                         save_dir='./models'):
    """
    Rolling window: train on first `block` runs, then for each subsequent block:
      - MC-predict with uncertainty bands and plot (MSE in top-center)
      - Also plot current profile on secondary y-axis
      - Save checkpoint named by last run’s date
      - Reload fresh model (reset optimizer/LR), load weights
      - Fine-tune on that block
    Finally, plots mean metrics per iteration.
    """
    os.makedirs(save_dir, exist_ok=True)
    dataset = BatteryRunDataset.from_directory(npz_dir)
    if dataset.removed_runs:
        print("Removed neg-current runs:", list(dataset.removed_runs))
    files = dataset.files
    run_by_name = {run.file_name: run for run in dataset.runs}
    N     = len(files)
    print(f"{N} total runs, block size = {block}")

    # 1) initial training
    init_files = files[:block]
    X0, Y0, _ = dataset.rolling_block(init_files).load_arrays(seq_len)
    args0 = copy.copy(args); args0.batch_size = block; args0.train_runs = block
    model = train(args0, X0, Y0, model_name=args0.model_name)

    iteration_metrics = []
    per_run_metrics = []

    # 2) rolling window
    for itr, start in enumerate(range(block, N, block), start=1):
        test_files = files[start:start+block]
        print(f"\nIteration {itr}: predicting runs {start}–{start+len(test_files)-1}")

        block_metrics = []
        dates_block  = []

        # 2a) MC-predict + plot each run
        for j, fn in enumerate(test_files, start=1):
            run = run_by_name[fn]
            curr = run.current
            volt = run.voltage
            date = run.date
            dates_block.append(date)

            inp = torch.from_numpy(curr).to(args.device).view(1, -1)
            prediction = predict_with_uncertainty(model, inp, mc_samples=mc_samples)
            mean_pred = prediction['mean']
            lower_pred = prediction['lower']
            upper_pred = prediction['upper']

            metrics_payload = compute_error_metrics(volt, mean_pred)
            met = [metrics_payload['mae'], metrics_payload['mape'], metrics_payload['mse'], metrics_payload['rmse']]
            block_metrics.append(met)

            # plot
            fig, axes = plot_rolling_prediction(
                volt,
                mean_pred,
                curr=curr,
                lower=lower_pred,
                upper=upper_pred,
                title=f"Iter {itr} Run {j}: {fn}",
                show_current_on_secondary_axis=True,
            )
            axes[0].text(0.5, 0.95,
                         f"MSE = {met[2]:.4e}",
                         transform=axes[0].transAxes,
                         ha='center', va='top',
                         fontsize=10,
                         bbox=dict(boxstyle="round,pad=0.3",
                                   fc="white", ec="black", alpha=0.7))
            fig.tight_layout()
            fig.show()

        # summarize iteration
        block_metrics = np.array(block_metrics)
        per_run_metrics.append(block_metrics)
        mean_metrics  = block_metrics.mean(axis=0)
        names = ['MAE','MAPE','MSE','RMSE']
        print(f"Iteration {itr} summary:")
        for n, val in zip(names, mean_metrics):
            print(f"  {n} = {val:.4e}")
        iteration_metrics.append(mean_metrics)

        # 2b) save checkpoint named by last run date
        last_date = dates_block[-1]
        safe_date = last_date.replace(' ', '_').replace(':','-')
        ckpt_path = os.path.join(save_dir, f"{model_name}_{safe_date}.pth")
        save_battery_checkpoint(model, save_path=ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # 2c) reload fresh model to reset optimizer/LR
        if model_name == 'BattNN':
            new_model = BattNN(args0)
        else:
            new_model = LSTM(args0)
        load_battery_checkpoint(new_model, ckpt_path, map_location=args.device)
        model = new_model.to(args.device)

        # 2d) fine-tune on this block
        Xf, Yf, _ = dataset.rolling_block(test_files).load_arrays(seq_len)
        argsf = copy.copy(args); argsf.batch_size = Xf.shape[0]; argsf.train_runs = Xf.shape[0]
        model.config = argsf
        model.set_batch_size(argsf.batch_size)
        model, _ = fine_tune_battery_model(model, Xf, Yf, argsf)

    # 3) plot iteration metrics
    iteration_metrics = np.stack(iteration_metrics)  # [n_iters, 4]
    its = np.arange(1, iteration_metrics.shape[0]+1)
    plt.figure(figsize=(8,4))
    for k, n in enumerate(['MAE','MAPE','MSE','RMSE']):
        plt.plot(its, iteration_metrics[:,k], '-o', label=n)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Metrics per Rolling Iteration')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return iteration_metrics, per_run_metrics



def main():
    args = get_args()
    from battery import build_rolling_experiment_config
    config = build_rolling_experiment_config(args, mode='variational')
    namespace = config.to_namespace()
    iter_metrics, run_metrics = rolling_fine_tune_uq(
        namespace.npz_dir,
        namespace.seq_len,
        block=namespace.block_size,
        model_name=namespace.model_name,
        args=namespace,
        mc_samples=namespace.mc_samples,
        save_dir=namespace.save_dir,
    )
    all_runs = np.vstack(run_metrics)
    np.savez(
        f'results/RollingVBattNN-{namespace.save_model}-batch_size={namespace.train_runs}-seq_len={namespace.seq_len}.npz',
        all_runs,
    )


if __name__ == '__main__':
    main()
