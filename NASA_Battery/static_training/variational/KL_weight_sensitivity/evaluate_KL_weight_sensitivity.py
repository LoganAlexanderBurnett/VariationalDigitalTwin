from __future__ import annotations

from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import (
    StaticExperimentDefaults,
    build_static_experiment_config,
    parse_args,
    run_static_experiment,
)
from battery.models import VariationalBatteryModel as BattNN


DEFAULTS = StaticExperimentDefaults(
    model_module_name="vBattNN",
    mode="variational",
    train_runs=300,
    seq_len=30,
    epoch=500,
)
KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _build_case_namespace(base_namespace, case_dir: Path, kl_weight: float):
    namespace = type(base_namespace)(**vars(base_namespace))
    namespace.kl_beta = float(kl_weight)
    namespace.results_dir = str(case_dir)
    return namespace


def main():
    args = parse_args(DEFAULTS)
    config = build_static_experiment_config(args)
    base_namespace = config.to_namespace()

    results_root = Path(__file__).resolve().parent

    for kl_weight in KL_WEIGHTS:
        case_dir = results_root / f"KL_{kl_weight:.0e}"
        case_dir.mkdir(parents=True, exist_ok=True)

        namespace = _build_case_namespace(base_namespace, case_dir=case_dir, kl_weight=kl_weight)

        start = time.time()
        mean_error, _ = run_static_experiment(namespace, model_cls=BattNN)
        elapsed = time.time() - start

        metrics = {
            "kl_weight": kl_weight,
            "mae": float(mean_error[0]),
            "mape": float(mean_error[1]),
            "mse": float(mean_error[2]),
            "rmse": float(mean_error[3]),
            "runtime_sec": elapsed,
        }
        pd.DataFrame([metrics]).to_csv(case_dir / "metrics.csv", index=False)

        print(f"Completed KL={kl_weight:.0e}; metrics saved to {case_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
