from pathlib import Path
import sys

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
from battery.models import DeterministicBatteryModel as BattNN


DEFAULTS = StaticExperimentDefaults(
    model_module_name="BattNN",
    mode="deterministic",
    train_runs=150,
    seq_len=30,
    epoch=2000,
)


def main():
    args = parse_args(DEFAULTS)
    config = build_static_experiment_config(args)
    namespace = config.to_namespace()
    return run_static_experiment(namespace, model_cls=BattNN)


if __name__ == "__main__":
    main()
