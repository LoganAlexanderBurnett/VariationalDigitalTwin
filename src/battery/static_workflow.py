from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch

from .data import BatteryRun, BatteryRunDataset
from .evaluation import compute_error_metrics, plot_static_prediction
from .predict import predict_deterministic_run, predict_with_uncertainty
from .trainer import (
    load_battery_checkpoint,
    train_battery_deterministic,
    train_battery_variational,
)


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


def describe_removed_runs(dataset: BatteryRunDataset) -> None:
    if dataset.removed_runs:
        print("Removed runs due to negative current:", list(dataset.removed_runs))
    else:
        print("No runs removed due to negative current.")


def eval_metrics(y_true, y_pred):
    metrics_payload = compute_error_metrics(y_true, y_pred)
    return [
        metrics_payload['mae'],
        metrics_payload['mape'],
        metrics_payload['mse'],
        metrics_payload['rmse'],
    ]






def build_static_checkpoint_path(args) -> Path:
    return Path(args.results_dir) / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}.pkl"
    )


def build_static_predictions_pdf_path(args) -> Path:
    return Path(args.results_dir) / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}-predictions.pdf"
    )


def train_static_model(args, train_x, train_y, model_cls):
    model = model_cls(args)
    print(f"Selected model: {args.model_name}")
    print(f"Training on {train_x.shape[0]} runs")

    checkpoint_path = build_static_checkpoint_path(args)
    if args.mode == "variational":
        model, _ = train_battery_variational(model, train_x, train_y, args, save_path=checkpoint_path)
    else:
        model, _ = train_battery_deterministic(model, train_x, train_y, args, save_path=checkpoint_path)
    return model




def evaluate_static_model(args, test_runs: Iterable[BatteryRun], model_cls):
    model = model_cls(args)
    load_battery_checkpoint(model, build_static_checkpoint_path(args), map_location=args.device)
    model.set_batch_size(1)

    errors = []
    predictions_pdf_path = build_static_predictions_pdf_path(args)
    predictions_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(predictions_pdf_path) as pdf:
        for run in test_runs:
            curr = run.current
            volt = run.voltage
            date = run.date
            current_tensor = torch.from_numpy(curr.astype(np.float32)).view(1, -1)
            if args.mode == "variational":
                current_tensor = current_tensor.to(args.device)
                prediction = predict_with_uncertainty(
                    model, current_tensor, mc_samples=args.mc_samples, n_jobs=args.n_jobs
                )
                pred = prediction['mean']
                lower = prediction['lower']
                upper = prediction['upper']
            else:
                pred, _ = predict_deterministic_run(model, current_tensor)
                lower = upper = None

            assert pred.shape[0] == volt.shape[0], (
                f"Length mismatch: pred={pred.shape[0]}, true={volt.shape[0]}"
            )

            metrics_payload = compute_error_metrics(volt, pred)
            errors.append([metrics_payload['mae'], metrics_payload['mape'], metrics_payload['mse'], metrics_payload['rmse']])

            fig, _, ax2 = plot_static_prediction(curr, volt, pred, date, lower=lower, upper=upper)
            ax2.text(
                0.5,
                0.95,
                f"MSE = {metrics_payload['mse']:.4e}",
                transform=ax2.transAxes,
                ha="center",
                va="top",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7),
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    errors = np.array(errors)
    mean_err = errors.mean(axis=0)
    print(f"Testing on {errors.shape[0]} runs")
    print("Saved prediction plots to", predictions_pdf_path)
    print("Test error [MAE, MAPE, MSE, RMSE]:", mean_err.tolist())
    return mean_err, errors




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
    parser.add_argument("--train-runs", "--batch_size", "-n", dest="train_runs", type=int, default=defaults.train_runs)
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
    dataset = BatteryRunDataset.from_directory(args.npz_dir)
    describe_removed_runs(dataset)
    split = dataset.static_split(train_count=args.train_runs)
    train_x, train_y, train_dates = split.load_train_arrays(length=args.seq_len)
    print("Train shape:", train_x.shape, train_y.shape)
    print("Train dates:", train_dates)

    train_static_model(args, train_x, train_y, model_cls=model_cls)
    mean_error, errors = evaluate_static_model(
        args,
        test_runs=split.iter_test_runs(),
        model_cls=model_cls,
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}.npz"
    )
    np.savez(result_path, errors=errors)
    print("Saved errors to", result_path)
    print("Test error [MAE, MAPE, MSE, RMSE]:", mean_error)
    return mean_error, errors
