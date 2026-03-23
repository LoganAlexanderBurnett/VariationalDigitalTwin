from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import StaticExperimentDefaults, parse_args, run_static_experiment
from vBattNN import BattNN


DEFAULTS = StaticExperimentDefaults(
    model_module_name="vBattNN",
    mode="variational",
    train_runs=150,
    seq_len=30,
    epoch=500,
)


def main():
    args = parse_args(DEFAULTS)
    return run_static_experiment(args, model_cls=BattNN)


if __name__ == "__main__":
    main()
