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

from battery import (  # noqa: E402
    BatteryRunDataset,
    StaticExperimentDefaults,
    build_static_experiment_config,
    load_battery_checkpoint,
    parse_args,
    predict_with_uncertainty,
    train_static_model,
)
from battery.models import VariationalBatteryModel as BattNN  # noqa: E402


DEFAULTS = StaticExperimentDefaults(
    model_module_name="vBattNN",
    mode="variational",
    train_runs=500,
    seq_len=30,
    epoch=500,
    npz_dir="../../../dataset/",
)
N_JOBS_CANDIDATES = [1, 3, 6, 12]
MAX_TEST_RUNS = 25


def _resolve_npz_dir(npz_dir: str) -> str:
    npz_path = Path(npz_dir)
    if npz_path.is_absolute():
        return str(npz_path)
    return str((Path(__file__).resolve().parent / npz_path).resolve())


def _build_checkpoint_path(args) -> Path:
    return Path(args.results_dir) / (
        f"{args.results_prefix}-{args.save_model}-batch_size={args.train_runs}-seq_len={args.seq_len}.pkl"
    )


def _seed_everything():
    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)


def _compute_summary(n_jobs: int, per_run_df: pd.DataFrame) -> dict:
    run_times = per_run_df["inference_sec"].to_numpy()
    total_points = int(per_run_df["n_points"].sum())
    total_time = float(run_times.sum())

    summary = {
        "n_jobs": int(n_jobs),
        "n_runs": int(len(per_run_df)),
        "n_points_total": total_points,
        "total_inference_sec": total_time,
        "avg_run_inference_sec": float(run_times.mean()),
        "median_run_inference_sec": float(np.median(run_times)),
        "p95_run_inference_sec": float(np.percentile(run_times, 95)),
        "min_run_inference_sec": float(run_times.min()),
        "max_run_inference_sec": float(run_times.max()),
        "std_run_inference_sec": float(run_times.std(ddof=0)),
        "overall_points_per_sec": float(total_points / total_time) if total_time > 0 else float("nan"),
        "avg_points_per_sec_per_run": float(per_run_df["points_per_sec"].mean()),
        "first_run_inference_sec": float(run_times[0]),
    }

    if len(run_times) > 1:
        steady_state = run_times[1:]
        summary["steady_state_avg_run_inference_sec"] = float(steady_state.mean())
        summary["steady_state_p95_run_inference_sec"] = float(np.percentile(steady_state, 95))
    else:
        summary["steady_state_avg_run_inference_sec"] = float("nan")
        summary["steady_state_p95_run_inference_sec"] = float("nan")

    return summary


def main():
    args = parse_args(DEFAULTS)
    args.npz_dir = _resolve_npz_dir(args.npz_dir)

    config = build_static_experiment_config(args)
    namespace = config.to_namespace()

    output_root = Path(__file__).resolve().parent / "results"
    output_root.mkdir(parents=True, exist_ok=True)
    namespace.results_dir = str(output_root)

    _seed_everything()
    print("Arguments:", namespace)

    dataset = BatteryRunDataset.from_directory(namespace.npz_dir)
    split = dataset.static_split(train_count=namespace.train_runs)
    train_x, train_y, train_dates = split.load_train_arrays(length=namespace.seq_len)
    all_test_runs = list(split.iter_test_runs())
    test_runs = all_test_runs[:MAX_TEST_RUNS]

    print("Train shape:", train_x.shape, train_y.shape)
    print("Train dates:", train_dates)
    print(f"Number of test runs selected for benchmarking: {len(test_runs)} / {len(all_test_runs)}")

    train_start = time.perf_counter()
    train_static_model(namespace, train_x, train_y, model_cls=BattNN)
    train_elapsed = time.perf_counter() - train_start
    checkpoint_path = _build_checkpoint_path(namespace)
    print(f"Training complete in {train_elapsed:.3f}s")
    print(f"Checkpoint: {checkpoint_path}")

    summary_records: list[dict] = []

    for n_jobs in N_JOBS_CANDIDATES:
        print("\n" + "=" * 80)
        print(f"Benchmarking inference with n_jobs={n_jobs}")

        model = BattNN(namespace)
        load_battery_checkpoint(model, checkpoint_path, map_location=namespace.device)
        model.set_batch_size(1)

        per_run_records = []
        run_counter = 0

        for run in test_runs:
            run_counter += 1
            current_tensor = torch.from_numpy(run.current.astype(np.float32)).view(1, -1).to(namespace.device)

            start = time.perf_counter()
            predict_with_uncertainty(
                model,
                current_tensor,
                mc_samples=namespace.mc_samples,
                n_jobs=n_jobs,
            )
            elapsed = time.perf_counter() - start

            n_points = int(len(run.current))
            per_run_records.append(
                {
                    "n_jobs": n_jobs,
                    "run_index": run_counter,
                    "date": run.date,
                    "n_points": n_points,
                    "inference_sec": elapsed,
                    "points_per_sec": (n_points / elapsed) if elapsed > 0 else float("nan"),
                }
            )

            if run_counter % 5 == 0:
                print(f"[n_jobs={n_jobs}] Completed {run_counter}/{len(test_runs)} test runs")

        per_run_df = pd.DataFrame(per_run_records)
        summary = _compute_summary(n_jobs=n_jobs, per_run_df=per_run_df)
        summary["train_time_sec"] = float(train_elapsed)
        summary_records.append(summary)

        per_run_path = output_root / f"inference_run_timings_n_jobs_{n_jobs}.csv"
        per_run_df.to_csv(per_run_path, index=False)
        print(f"Saved per-run timings to: {per_run_path}")
        print(
            f"n_jobs={n_jobs} | total={summary['total_inference_sec']:.3f}s | "
            f"avg/run={summary['avg_run_inference_sec']:.3f}s | "
            f"p95/run={summary['p95_run_inference_sec']:.3f}s"
        )

    summary_df = pd.DataFrame(summary_records).sort_values("n_jobs").reset_index(drop=True)
    summary_path = output_root / "inference_speed_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nSaved inference speed summary to:", summary_path)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
