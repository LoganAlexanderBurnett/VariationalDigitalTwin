import os
import copy
import random
import argparse
from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from BattNN import BattNN

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import BatteryRunDataset

# reproducibility
random.seed(2022)
np.random.seed(2022)
torch.manual_seed(2022)

def eval_metrics(y_true, y_pred):
    MAE  = metrics.mean_absolute_error     (y_true, y_pred)
    MAPE = metrics.mean_absolute_percentage_error(y_true, y_pred)
    MSE  = metrics.mean_squared_error      (y_true, y_pred)
    RMSE = np.sqrt(MSE)
    return [MAE, MAPE, MSE, RMSE]

def train(args, train_x, train_y, model_name='BattNN'):
    # 1) Create model
    if model_name == 'BattNN':
        model = BattNN(args)
        print("Selected model: BattNN")
    elif model_name == 'LSTM':
        model = LSTM(args)
        print("Selected model: LSTM")
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 2) Package data into the model
    x_tensor = torch.from_numpy(train_x.astype(np.float32))
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    model.get_data(x=x_tensor, label=y_tensor)

    # 3) Train
    model.train()
    return model  # in case you want to keep it around
    

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
    parser.add_argument('--batch_size', '-n', type=int,   default=30)
    parser.add_argument('--seq_len',    '-l', type=int,   default=30)
    parser.add_argument('--npz_dir',              default='../../dataset/')

    # — Optimization params —
    parser.add_argument('--device',     default='cpu')
    parser.add_argument('--epoch',      type=int,   default=2000)
    parser.add_argument('--lr',         type=float, default=2e-2)
    parser.add_argument('--weight_decay',type=float,default=5e-4)

    # — Model selection & saving —
    parser.add_argument('--model_name', choices=['BattNN','LSTM'], default='BattNN')
    parser.add_argument('--save_model', choices=[None,'NASA'], default='NASA')

    # ignore Jupyter args
    args, _ = parser.parse_known_args()
    return args

# ── Monte-Carlo predict without invoking your override of train() ─────────────
def mc_predict(model, current_seq, mc_samples=100):
    """
    Perform MC forward passes to get mean & std of voltage predictions,
    without calling model.eval() (which would break BattNN.train signature).
    """
    # temporarily set all submodules to eval mode via the base class method
    base_train = torch.nn.Module.train
    # switch off training behaviors
    base_train(model, False)
    preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            v_pred, _ = model.predict(current_seq)   # [1, T]
            preds.append(v_pred.cpu().numpy()[0])
    # restore training mode
    base_train(model, True)

    stack = np.stack(preds, axis=0)  # [mc_samples, T]
    return stack.mean(axis=0), stack.std(axis=0)

# ── Rolling UQ digital‐twin demo ───────────────────────────────────────────────
def rolling_fine_tune_uq(npz_dir, seq_len, block=5,
                         model_name='BattNN', args=None,
                         mc_samples=100,
                         save_dir='./models'):
    """
    Rolling window: train on first `block` runs, then for each subsequent block:
      - MC-predict with uncertainty bands and plot (MSE in top-center)
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
    args0 = copy.copy(args); args0.batch_size = block
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
            mean_pred, std_pred = mc_predict(model, inp, mc_samples=mc_samples)

            met = eval_metrics(volt, mean_pred)
            block_metrics.append(met)

            # plot
            t = np.arange(len(volt))
            fig, ax = plt.subplots(figsize=(8,3))
            ax.plot(t, volt,      '--k', label='True')
            ax.plot(t, mean_pred, 'b-',  label='Mean pred')
            ax.fill_between(t,
                            mean_pred - std_pred,
                            mean_pred + std_pred,
                            alpha=0.3, color='b', label='±σ')
            # MSE annotation at top-center
            ax.text(0.5, 0.95,
                    f"MSE = {met[2]:.4e}",
                    transform=ax.transAxes,
                    ha='center', va='top',
                    fontsize=12,
                    bbox=dict(fc="white", alpha=0.7))
            ax.set_xlabel('Time step')
            ax.set_ylabel('Voltage (V)')
            ax.legend(loc='upper right')
            plt.title(f"Iter {itr} Run {j}: {fn}")
            plt.tight_layout()
            plt.show()

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
        ckpt = {'net': model.state_dict()}
        ckpt_path = os.path.join(save_dir,
                                 f"{model_name}_{safe_date}.pth")
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # 2c) reload fresh model to reset optimizer/LR
        if model_name == 'BattNN':
            new_model = BattNN(args0)
        else:
            new_model = LSTM(args0)
        new_model.load_state_dict(torch.load(ckpt_path)['net'])
        model = new_model.to(args.device)

        # 2d) fine-tune on this block
        Xf, Yf, _ = dataset.rolling_block(test_files).load_arrays(seq_len)
        argsf = copy.copy(args); argsf.batch_size = Xf.shape[0]
        model.config = argsf
        model.init_x = torch.tensor(
            argsf.x0, dtype=model.init_x.dtype,
            device=model.init_x.device
        ).repeat(argsf.batch_size, 1)
        model.get_data(torch.from_numpy(Xf), torch.from_numpy(Yf))
        model.train()

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

args = get_args()
iter_metrics, run_metrics = rolling_fine_tune_uq(
    args.npz_dir, args.seq_len,
    block=10,
    model_name=args.model_name,
    args=args,
    mc_samples=1
)

all_runs = np.vstack(run_metrics)
all_runs.shape
np.savez(f'results/RollingBattNN-{args.save_model}-batch_size={args.batch_size}-seq_len={args.seq_len}.npz', all_runs)