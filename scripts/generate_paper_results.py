#!/usr/bin/env python3
"""Generate a consolidated `paper_results/` directory for reproducible paper artifacts."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_RESULTS_DIR = REPO_ROOT / "paper_results"

PLOT_SCRIPTS = [
    REPO_ROOT / "NASA_Battery" / "plot_static_vs_rolling.py",
    REPO_ROOT / "HTTF" / "static_training" / "plot_model_comparisons.py",
]

ARTIFACT_PATTERNS = [
    # NASA battery figures and summary outputs
    "NASA_Battery/BattNNStaticVsRolling.png",
    "NASA_Battery/BattNNStaticVsRolling.csv",
    "NASA_Battery/battNN_viz.png",
    "NASA_Battery/static_training/deterministic/results/*.npz",
    "NASA_Battery/static_training/deterministic/results/*predictions.pdf",
    "NASA_Battery/static_training/variational/results/*.npz",
    "NASA_Battery/static_training/variational/results/*predictions.pdf",
    "NASA_Battery/rolling_training/variational/results/*.npz",
    "NASA_Battery/rolling_training/variational/results/*predictions.pdf",
    # HTTF figures
    "HTTF/static_training/compare_*_*.png",
    "HTTF/static_training/deterministic/*/performance_metrics.json",
    "HTTF/static_training/variational/*/performance_metrics.json",
    # PSML figures
    "PSML/static_training/variational/LSTM_GRU_PSML_*.png",
    "PSML/rolling_training/variational/PSML_rolling_metrics.png",
    "PSML/psml_viz.png",
    # HTTF shrinking-trainset summary figure
    "HTTF/shrinking_trainset/variational/HTTF_shrink_metrics.png",
]


def run_plot_scripts() -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    for script_path in PLOT_SCRIPTS:
        if not script_path.exists():
            print(f"[WARN] Skipping missing plot script: {script_path}")
            continue
        print(f"[INFO] Running plot script: {script_path.relative_to(REPO_ROOT)}")
        try:
            subprocess.run([sys.executable, str(script_path)], cwd=REPO_ROOT, env=env, check=True)
        except (subprocess.CalledProcessError, ModuleNotFoundError) as exc:
            print(f"[WARN] Plot script failed and will be skipped: {script_path.relative_to(REPO_ROOT)} :: {exc}")


def collect_artifacts() -> list[Path]:
    copied: list[Path] = []
    for pattern in ARTIFACT_PATTERNS:
        for source in REPO_ROOT.glob(pattern):
            if not source.is_file():
                continue
            relative_source = source.relative_to(REPO_ROOT)
            destination = PAPER_RESULTS_DIR / relative_source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            copied.append(relative_source)
    return sorted(set(copied))


def write_manifest(copied_files: list[Path]) -> None:
    manifest_path = PAPER_RESULTS_DIR / "MANIFEST.md"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Results Manifest",
        "",
        "This directory is generated automatically by `scripts/generate_paper_results.py`.",
        "",
        f"Total copied artifacts: **{len(copied_files)}**",
        "",
        "## Included files",
        "",
    ]
    for path in copied_files:
        lines.append(f"- `{path.as_posix()}`")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate consolidated paper_results artifacts")
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip executing plot scripts and only copy existing outputs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete existing paper_results directory before collecting outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.clean and PAPER_RESULTS_DIR.exists():
        print(f"[INFO] Removing existing directory: {PAPER_RESULTS_DIR}")
        shutil.rmtree(PAPER_RESULTS_DIR)

    if not args.skip_plots:
        run_plot_scripts()

    copied_files = collect_artifacts()
    write_manifest(copied_files)

    print(f"[INFO] Paper results are available in: {PAPER_RESULTS_DIR}")
    print(f"[INFO] Copied {len(copied_files)} artifact(s).")


if __name__ == "__main__":
    main()
