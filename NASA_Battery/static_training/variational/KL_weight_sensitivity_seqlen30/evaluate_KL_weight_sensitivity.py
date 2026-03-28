from __future__ import annotations

from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import (
    BatteryRunDataset,
    StaticExperimentDefaults,
    build_static_experiment_config,
    evaluate_static_model,
    load_battery_checkpoint,
    parse_args,
    predict_with_uncertainty,
    train_static_model,
)
from battery.models import VariationalBatteryModel as BattNN
from battery.static_workflow import describe_removed_runs


DEFAULTS = StaticExperimentDefaults(
    model_module_name="vBattNN",
    mode="variational",
    train_runs=500,
    seq_len=30,
    epoch=500,
    npz_dir="../../../dataset/",
)
KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _resolve_npz_dir(npz_dir: str) -> str:
    npz_path = Path(npz_dir)
    if npz_path.is_absolute():
        return str(npz_path)
    return str((Path(__file__).resolve().parent / npz_path).resolve())


def _build_case_namespace(base_namespace, case_dir: Path, kl_weight: float):
    namespace = type(base_namespace)(**vars(base_namespace))
    namespace.kl_beta = float(kl_weight)
    namespace.results_dir = str(case_dir)
    return namespace


def _build_checkpoint_path(args) -> Path:
    return Path(args.results_dir) / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}.pkl"
    )


def _save_per_point_inference_csv(args, split, model_cls, output_csv: Path):
    model = model_cls(args)
    load_battery_checkpoint(model, _build_checkpoint_path(args), map_location=args.device)
    model.set_batch_size(1)

    records = []
    for run_idx, run in enumerate(split.iter_test_runs(), start=1):
        current_tensor = torch.from_numpy(run.current.astype(np.float32)).view(1, -1).to(args.device)
        prediction = predict_with_uncertainty(
            model,
            current_tensor,
            mc_samples=args.mc_samples,
            n_jobs=args.n_jobs,
        )

        mean_pred = prediction["mean"]
        lower = prediction["lower"]
        upper = prediction["upper"]
        true_voltage = run.voltage

        for time_idx, (y_true, y_pred_mean, y_lower, y_upper) in enumerate(
            zip(true_voltage, mean_pred, lower, upper), start=1
        ):
            records.append(
                {
                    "run_index": run_idx,
                    "date": run.date,
                    "time_index": time_idx,
                    "true": float(y_true),
                    "pred_mean": float(y_pred_mean),
                    "lower_ci": float(y_lower),
                    "upper_ci": float(y_upper),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output_csv, index=False)
    print(f"Saved per-point inference data to {output_csv}")


def _run_static_experiment_with_timing(args, model_cls):
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

    train_start = time.time()
    train_static_model(args, train_x, train_y, model_cls=model_cls)
    train_time = time.time() - train_start

    infer_start = time.time()
    mean_error, errors = evaluate_static_model(
        args,
        test_runs=split.iter_test_runs(),
        model_cls=model_cls,
    )
    infer_time = time.time() - infer_start

    _save_per_point_inference_csv(
        args,
        split,
        model_cls=model_cls,
        output_csv=Path(args.results_dir) / "inference_predictions.csv",
    )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}.npz"
    )
    np.savez(result_path, errors=errors)
    print("Saved errors to", result_path)
    print("Test error [MAE, MAPE, MSE, RMSE]:", mean_error)

    return mean_error, {
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "runtime_sec": train_time + infer_time,
    }


def main():
    args = parse_args(DEFAULTS)
    args.npz_dir = _resolve_npz_dir(args.npz_dir)

    config = build_static_experiment_config(args)
    base_namespace = config.to_namespace()

    results_root = Path(__file__).resolve().parent

    for kl_weight in KL_WEIGHTS:
        case_dir = results_root / f"KL_{kl_weight:.0e}"
        case_dir.mkdir(parents=True, exist_ok=True)

        namespace = _build_case_namespace(base_namespace, case_dir=case_dir, kl_weight=kl_weight)

        mean_error, timing = _run_static_experiment_with_timing(namespace, model_cls=BattNN)

        metrics = {
            "kl_weight": kl_weight,
            "mae": float(mean_error[0]),
            "mape": float(mean_error[1]),
            "mse": float(mean_error[2]),
            "rmse": float(mean_error[3]),
            **timing,
        }
        pd.DataFrame([metrics]).to_csv(case_dir / "metrics.csv", index=False)

        print(f"Completed KL={kl_weight:.0e}; metrics saved to {case_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
