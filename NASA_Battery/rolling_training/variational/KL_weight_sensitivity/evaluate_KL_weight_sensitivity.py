from __future__ import annotations

import argparse
from copy import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import (  # noqa: E402
    BatteryRunDataset,
    build_rolling_experiment_config,
    compute_error_metrics,
    fine_tune_battery_model,
    load_battery_checkpoint,
    predict_with_uncertainty,
    save_battery_checkpoint,
)
from battery.models import VariationalBatteryModel as BattNN  # noqa: E402


KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
RESULTS_ROOT = Path(__file__).resolve().parent


def get_args():
    parser = argparse.ArgumentParser(description="Rolling KL-weight sensitivity for variational BattNN")
    parser.add_argument("--V0", type=float, default=4.2)
    parser.add_argument("--x0", nargs=3, type=float, default=[8000, 0, 0])
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--VEOD", type=float, default=3.2)
    parser.add_argument("--Rp", type=float, default=1000)
    parser.add_argument("--Rs", type=float, default=0.5)
    parser.add_argument("--Csp", type=float, default=15)
    parser.add_argument("--Cs", type=float, default=500)
    parser.add_argument("--train-runs", "--batch_size", "-n", dest="train_runs", type=int, default=30)
    parser.add_argument("--seq_len", "-l", type=int, default=30)
    parser.add_argument("--npz_dir", default="../../../dataset/")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--model_name", choices=["BattNN", "LSTM"], default="BattNN")
    parser.add_argument("--save_model", default="NASA")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--save-dir", default="./models")
    parser.add_argument("--mc_samples", type=int, default=100)
    parser.add_argument("--n_jobs", type=int, default=10)
    args, _ = parser.parse_known_args()
    return args


def _resolve_npz_dir(npz_dir: str) -> str:
    npz_path = Path(npz_dir)
    if npz_path.is_absolute():
        return str(npz_path)
    return str((Path(__file__).resolve().parent / npz_path).resolve())


def _build_model(args):
    if args.model_name != "BattNN":
        raise ValueError(f"Unsupported model: {args.model_name}")
    return BattNN(args)


def _compute_uncertainty_metrics(true_values: np.ndarray, mean_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    abs_error = np.abs(true_values - mean_pred)
    interval_width = upper - lower
    coverage = float(((true_values >= lower) & (true_values <= upper)).mean())
    spearman_corr = float(pd.Series(interval_width).corr(pd.Series(abs_error), method="spearman"))
    return {
        "coverage": coverage,
        "spearman_corr_uncertainty_error": spearman_corr,
    }


def _save_json(data, filename: Path):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run_rolling_for_kl(args, kl_weight: float):
    case_dir = RESULTS_ROOT / f"KL_{kl_weight:.0e}"
    case_dir.mkdir(parents=True, exist_ok=True)
    models_dir = case_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    args.results_dir = str(case_dir)
    args.save_dir = str(models_dir)
    args.kl_beta = float(kl_weight)

    dataset = BatteryRunDataset.from_directory(args.npz_dir)
    if dataset.removed_runs:
        print("Removed neg-current runs:", list(dataset.removed_runs))

    files = dataset.files
    run_by_name = {run.file_name: run for run in dataset.runs}

    block_size = int(args.block_size)
    init_files = files[:block_size]

    train_x, train_y, _ = dataset.rolling_block(init_files).load_arrays(args.seq_len)
    init_args = copy(args)
    init_args.batch_size = block_size
    init_args.train_runs = block_size

    model = _build_model(init_args)
    model, _ = fine_tune_battery_model(model, train_x, train_y, init_args)

    session_rows = []
    all_true, all_pred, all_lower, all_upper = [], [], [], []
    train_times, infer_times = [], []

    for session, start in enumerate(range(block_size, len(files), block_size)):
        test_files = files[start:start + block_size]
        if not test_files:
            continue

        print(f"KL={kl_weight:.0e} | Session {session:03d}: {len(test_files)} run(s)")
        session_dir = case_dir / f"session_{session:03d}"
        session_dir.mkdir(parents=True, exist_ok=True)

        session_true, session_pred, session_lower, session_upper = [], [], [], []

        infer_start = time.perf_counter()
        for run_idx, file_name in enumerate(test_files, start=1):
            run = run_by_name[file_name]
            curr = run.current
            volt = run.voltage

            input_tensor = torch.from_numpy(curr.astype(np.float32)).to(args.device).view(1, -1)
            prediction = predict_with_uncertainty(model, input_tensor, mc_samples=args.mc_samples, n_jobs=args.n_jobs)

            mean_pred = prediction["mean"]
            lower = prediction["lower"]
            upper = prediction["upper"]

            session_true.append(volt)
            session_pred.append(mean_pred)
            session_lower.append(lower)
            session_upper.append(upper)

            if run_idx % 5 == 0:
                print(f"  Predicted {run_idx}/{len(test_files)} runs in session {session:03d}")

        infer_time = time.perf_counter() - infer_start
        infer_times.append(infer_time)

        true_arr = np.concatenate(session_true)
        pred_arr = np.concatenate(session_pred)
        lower_arr = np.concatenate(session_lower)
        upper_arr = np.concatenate(session_upper)

        base_metrics = compute_error_metrics(true_arr, pred_arr)
        unc_metrics = _compute_uncertainty_metrics(true_arr, pred_arr, lower_arr, upper_arr)

        session_metrics = {
            "session": session,
            "num_runs": len(test_files),
            "mae": float(base_metrics["mae"]),
            "mape": float(base_metrics["mape"]),
            "mse": float(base_metrics["mse"]),
            "rmse": float(base_metrics["rmse"]),
            "coverage": unc_metrics["coverage"],
            "spearman_corr_uncertainty_error": unc_metrics["spearman_corr_uncertainty_error"],
            "inference_time_seconds": float(infer_time),
        }
        session_rows.append(session_metrics)
        _save_json(session_metrics, session_dir / "metrics.json")

        session_pred_df = pd.DataFrame(
            {
                "True Voltage": true_arr,
                "Predicted Mean Voltage": pred_arr,
                "Lower CI Voltage": lower_arr,
                "Upper CI Voltage": upper_arr,
            }
        )
        session_pred_df.to_csv(session_dir / "vBattTest.csv", index=False)

        all_true.append(true_arr)
        all_pred.append(pred_arr)
        all_lower.append(lower_arr)
        all_upper.append(upper_arr)

        checkpoint_date = run_by_name[test_files[-1]].date
        safe_date = checkpoint_date.replace(" ", "_").replace(":", "-")
        checkpoint_path = models_dir / f"{args.model_name}_{safe_date}.pth"
        save_battery_checkpoint(model, save_path=checkpoint_path)

        next_args = copy(args)
        next_args.batch_size = len(test_files)
        next_args.train_runs = len(test_files)
        next_model = _build_model(next_args)
        load_battery_checkpoint(next_model, checkpoint_path, map_location=args.device)
        next_model = next_model.to(args.device)

        fine_tune_x, fine_tune_y, _ = dataset.rolling_block(test_files).load_arrays(args.seq_len)
        train_start = time.perf_counter()
        next_model.config = next_args
        next_model.set_batch_size(next_args.batch_size)
        model, _ = fine_tune_battery_model(next_model, fine_tune_x, fine_tune_y, next_args)
        train_times.append(time.perf_counter() - train_start)

    if not all_true:
        raise RuntimeError("No rolling sessions were executed. Check block-size and dataset size.")

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)
    lower_all = np.concatenate(all_lower)
    upper_all = np.concatenate(all_upper)

    base_case_metrics = compute_error_metrics(true_all, pred_all)
    unc_case_metrics = _compute_uncertainty_metrics(true_all, pred_all, lower_all, upper_all)

    case_metrics = {
        "kl_weight": kl_weight,
        "mae": float(base_case_metrics["mae"]),
        "mape": float(base_case_metrics["mape"]),
        "mse": float(base_case_metrics["mse"]),
        "rmse": float(base_case_metrics["rmse"]),
        "coverage": unc_case_metrics["coverage"],
        "spearman_corr_uncertainty_error": unc_case_metrics["spearman_corr_uncertainty_error"],
        "num_sessions": len(session_rows),
        "train_time_sec": float(np.sum(train_times)) if train_times else 0.0,
        "infer_time_sec": float(np.sum(infer_times)),
        "avg_train_time_sec_per_session": float(np.mean(train_times)) if train_times else 0.0,
        "avg_infer_time_sec_per_session": float(np.mean(infer_times)),
    }

    pd.DataFrame(session_rows).to_csv(case_dir / "session_metrics_summary.csv", index=False)
    pd.DataFrame([case_metrics]).to_csv(case_dir / "metrics.csv", index=False)
    pd.DataFrame(
        {
            "True Voltage": true_all,
            "Predicted Mean Voltage": pred_all,
            "Lower CI Voltage": lower_all,
            "Upper CI Voltage": upper_all,
        }
    ).to_csv(case_dir / "vBattTest.csv", index=False)

    run_config = {
        "kl_weight": kl_weight,
        "block_size": block_size,
        "mc_samples": args.mc_samples,
        "n_jobs": args.n_jobs,
        "device": str(args.device),
        "num_files": len(files),
        "num_sessions": len(session_rows),
    }
    _save_json(run_config, case_dir / "run_config.json")

    print(f"Completed KL={kl_weight:.0e}. Saved outputs to {case_dir}")


def main():
    raw_args = get_args()
    raw_args.npz_dir = _resolve_npz_dir(raw_args.npz_dir)

    config = build_rolling_experiment_config(raw_args, mode="variational")
    base_args = config.to_namespace()

    for kl_weight in KL_WEIGHTS:
        args = copy(base_args)
        run_rolling_for_kl(args, kl_weight)


if __name__ == "__main__":
    main()
