from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn import metrics


DEFAULT_DATE_FORMAT = "%d-%b-%Y %H:%M:%S"


@dataclass(frozen=True)
class StaticExperimentDefaults:
    model_module_name: str
    mode: str
    train_runs: int
    seq_len: int = 30
    npz_dir: str = "../../dataset/"
    plot_n: int = 3
    mc_samples: int = 100
    n_jobs: int = 6
    device: str = "cpu"
    epoch: Optional[int] = None
    lr: float = 2e-2
    weight_decay: float = 5e-4
    V0: float = 4.2
    x0: tuple[float, float, float] = (8000, 0, 0)
    dt: float = 1.0
    VEOD: float = 3.2
    Rp: float = 1000
    Rs: float = 0.5
    Csp: float = 15
    Cs: float = 500
    save_model: str = "NASA"

    def default_epoch(self) -> int:
        if self.epoch is not None:
            return self.epoch
        return 500 if self.mode == "variational" else 2000


class NPZData:
    def __init__(self, npz_dir: str = "../../dataset/", n: int = 50, length: int = 50, date_fmt: str = DEFAULT_DATE_FORMAT):
        self.npz_dir = npz_dir
        all_files = sorted(
            f for f in os.listdir(npz_dir) if f.startswith("run_") and f.endswith(".npz")
        )
        admitted = []
        removed = []

        for fname in all_files:
            data = np.load(os.path.join(npz_dir, fname))
            curr = np.asarray(data["current"])
            if np.any(curr < 0):
                removed.append(fname)
            else:
                admitted.append(fname)

        if removed:
            print("Removed runs due to negative current:", removed)
        else:
            print("No runs removed due to negative current.")

        dated_files = []
        for fname in admitted:
            data = np.load(os.path.join(npz_dir, fname))
            date_str = str(data["date"].item())
            dt = datetime.strptime(date_str, date_fmt)
            dated_files.append((fname, dt))

        dated_files.sort(key=lambda x: x[1])
        sorted_files = [fname for fname, _ in dated_files]

        self.train_files = sorted_files[:n]
        self.test_files = sorted_files[n:]
        self.length = length

    def load_train_data(self):
        currents, voltages, dates = [], [], []
        for fname in self.train_files:
            data = np.load(os.path.join(self.npz_dir, fname))
            curr = data["current"][: self.length].astype(np.float32)
            volt = data["voltage"][: self.length].astype(np.float32)
            date = str(data["date"].item())
            if curr.size == self.length and volt.size == self.length:
                currents.append(curr)
                voltages.append(volt)
                dates.append(date)
        return np.stack(currents), np.stack(voltages), dates

    def yield_test_data(self) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
        for fname in self.test_files:
            data = np.load(os.path.join(self.npz_dir, fname))
            curr = data["current"].astype(np.float32)
            volt = data["voltage"].astype(np.float32)
            date = str(data["date"].item())
            yield curr, volt, date


def eval_metrics(y_true, y_pred):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return [mae, mape, mse, rmse]


def train_static_model(args, train_x, train_y, model_cls):
    model = model_cls(args)
    print(f"Selected model: {args.model_name}")
    print(f"Training on {train_x.shape[0]} runs")

    x_tensor = torch.from_numpy(train_x.astype(np.float32))
    y_tensor = torch.from_numpy(train_y.astype(np.float32))
    model.get_data(x=x_tensor, label=y_tensor)
    model.train()
    return model


def _mc_predict(model, current_seq, mc_samples=100, n_jobs=6):
    base_train = torch.nn.Module.train
    base_train(model, False)

    def sample_prediction():
        with torch.no_grad():
            v_pred, _ = model.predict(current_seq)
        return v_pred.cpu().numpy()[0]

    preds = Parallel(n_jobs=n_jobs)(delayed(sample_prediction)() for _ in range(mc_samples))
    stack = np.stack(preds, axis=0)
    mean = stack.mean(axis=0)
    lower = np.percentile(stack, 2.5, axis=0)
    upper = np.percentile(stack, 97.5, axis=0)
    return mean, lower, upper


def evaluate_static_model(args, data_iter: Callable[[], Iterable[tuple[np.ndarray, np.ndarray, str]]], model_cls):
    model = model_cls(args)
    model.load_model()
    model.init_x = model.init_x[:1, :]

    errors = []
    for i, (curr, volt, date) in enumerate(data_iter(), start=1):
        current_tensor = torch.from_numpy(curr.astype(np.float32)).view(1, -1)
        if args.mode == "variational":
            current_tensor = current_tensor.to(args.device)
            pred, lower, upper = _mc_predict(
                model,
                current_tensor,
                mc_samples=args.mc_samples,
                n_jobs=args.n_jobs,
            )
        else:
            pred_tensor, _ = model.predict(current_tensor)
            pred = pred_tensor.detach().cpu().numpy().ravel()
            lower = upper = None

        assert pred.shape[0] == volt.shape[0], (
            f"Length mismatch: pred={pred.shape[0]}, true={volt.shape[0]}"
        )

        met = eval_metrics(volt, pred)
        errors.append(met)

        if i <= args.plot_n:
            _plot_prediction(curr, volt, pred, date, met[2], lower=lower, upper=upper)

    errors = np.array(errors)
    mean_err = errors.mean(axis=0)
    print(f"Testing on {errors.shape[0]} runs")
    print("Test error [MAE, MAPE, MSE, RMSE]:", mean_err.tolist())
    return mean_err, errors


def _plot_prediction(curr, volt, pred, date, mse_value, lower=None, upper=None):
    t = np.arange(len(volt))
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()

    ax1.plot(t, curr, color="g", label="Current")
    ax1.set_ylabel("Current (A)", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax1.set_xticks([0])
    ax1.set_xticklabels([date], rotation=45, ha="right")

    ax2.plot(t, volt, "-r", label="Voltage (true)")
    ax2.plot(t, pred, "-b" if lower is not None else "--r", label="Voltage (pred)" if lower is None else "Voltage (mean pred)")
    if lower is not None and upper is not None:
        ax2.fill_between(t, lower, upper, color="b", alpha=0.3, label="95% CI")
    ax2.set_ylabel("Voltage (V)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.text(
        0.5,
        0.95,
        f"MSE = {mse_value:.4e}",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
    )

    l1, lab1 = ax1.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, loc="upper right")

    plt.tight_layout()
    plt.show()


def build_static_arg_parser(defaults: StaticExperimentDefaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Battery Net for NPZData")
    parser.add_argument("--V0", type=float, default=defaults.V0)
    parser.add_argument("--x0", nargs=3, type=float, default=list(defaults.x0))
    parser.add_argument("--dt", type=float, default=defaults.dt)
    parser.add_argument("--VEOD", type=float, default=defaults.VEOD)
    parser.add_argument("--Rp", type=float, default=defaults.Rp)
    parser.add_argument("--Rs", type=float, default=defaults.Rs)
    parser.add_argument("--Csp", type=float, default=defaults.Csp)
    parser.add_argument("--Cs", type=float, default=defaults.Cs)
    parser.add_argument("--batch_size", "-n", type=int, default=defaults.train_runs)
    parser.add_argument("--seq_len", "-l", type=int, default=defaults.seq_len)
    parser.add_argument("--npz_dir", default=defaults.npz_dir)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--epoch", type=int, default=defaults.default_epoch())
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight_decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--model_name", choices=["BattNN", "LSTM"], default="BattNN")
    parser.add_argument("--save_model", default=defaults.save_model)
    parser.add_argument("--plot_n", type=int, default=defaults.plot_n)
    parser.add_argument("--mode", choices=["deterministic", "variational"], default=defaults.mode)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--results_prefix", default=defaults.model_module_name)
    parser.add_argument("--mc_samples", type=int, default=defaults.mc_samples)
    parser.add_argument("--n_jobs", type=int, default=defaults.n_jobs)
    return parser


def parse_args(defaults: StaticExperimentDefaults):
    parser = build_static_arg_parser(defaults)
    args, _ = parser.parse_known_args()
    return args


def run_static_experiment(args, model_cls):
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)

    print("Arguments:", args)
    data = NPZData(npz_dir=args.npz_dir, n=args.batch_size, length=args.seq_len)
    train_x, train_y, train_dates = data.load_train_data()
    print("Train shape:", train_x.shape, train_y.shape)
    print("Train dates:", train_dates)

    train_static_model(args, train_x, train_y, model_cls=model_cls)
    mean_error, errors = evaluate_static_model(
        args,
        data_iter=data.yield_test_data,
        model_cls=model_cls,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.batch_size}-seq_len={args.seq_len}.npz"
    )
    np.savez(result_path, errors=errors)
    print("Saved errors to", result_path)
    print("Test error [MAE, MAPE, MSE, RMSE]:", mean_error)
    return mean_error, errors
