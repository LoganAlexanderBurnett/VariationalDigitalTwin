import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "battery").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery import build_rolling_experiment_config, run_deterministic_rolling_experiment
from BattNN import BattNN


def get_args():
    parser = argparse.ArgumentParser(description='Battery Net for NPZData')
    parser.add_argument('--V0', type=float, default=4.2)
    parser.add_argument('--x0', nargs=3, type=float, default=[8000, 0, 0])
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--VEOD', type=float, default=3.2)
    parser.add_argument('--Rp', type=float, default=1000)
    parser.add_argument('--Rs', type=float, default=0.5)
    parser.add_argument('--Csp', type=float, default=15)
    parser.add_argument('--Cs', type=float, default=500)
    parser.add_argument('--train-runs', '--batch_size', '-n', dest='train_runs', type=int, default=30)
    parser.add_argument('--seq_len', '-l', type=int, default=30)
    parser.add_argument('--npz_dir', default='../../dataset/')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epoch', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-2)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--model_name', choices=['BattNN', 'LSTM'], default='BattNN')
    parser.add_argument('--save_model', choices=[None, 'NASA'], default='NASA')
    parser.add_argument('--block-size', type=int, default=10)
    parser.add_argument('--save-dir', default='./models')
    parser.add_argument('--mc_samples', type=int, default=1)
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    config = build_rolling_experiment_config(args, mode='deterministic')
    namespace = config.to_namespace()
    _, run_metrics = run_deterministic_rolling_experiment(
        model_cls=BattNN,
        args=namespace,
        block_size=namespace.block_size,
        mc_samples=namespace.mc_samples,
        save_dir=namespace.save_dir,
    )
    all_runs = np.vstack(run_metrics)
    np.savez(
        f'results/RollingBattNN-{namespace.save_model}-batch_size={namespace.train_runs}-seq_len={namespace.seq_len}.npz',
        all_runs,
    )


if __name__ == '__main__':
    main()
