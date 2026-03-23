from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

SHRINK_DIR_PREFIX = "shrink_train_"
METRIC_NAMES = ("r2", "mae", "mape", "rmse", "rmspe")
METRIC_SPECS = {
    "R²": {"key": "r2", "ylabel": r"$R^2$", "ylim": (0.5, 1.03)},
    "MAE": {"key": "mae", "ylabel": "MAE (°C)"},
    "RMSE": {"key": "rmse", "ylabel": "RMSE (°C)"},
}
MODEL_CONFIGS = {
    "vGRU": {
        "directory": "GRU",
        "marker": "o",
        "colors": {"TS": "dodgerblue", "TF": "mediumseagreen"},
    },
    "vLSTM": {
        "directory": "LSTM",
        "marker": "D",
        "colors": {"TS": "orangered", "TF": "goldenrod"},
    },
}
OUTPUT_CONFIGS = {
    "TS": {"metric_key": "output_1", "ypred_idx": 0},
    "TF": {"metric_key": "output_2", "ypred_idx": 1},
}


def iter_shrink_dirs(model_dir: Path) -> Iterable[tuple[int, Path]]:
    """Yield `(N_removed, path)` pairs sorted by removed sensor count."""
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {model_dir}")

    shrink_dirs: list[tuple[int, Path]] = []
    for path in model_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(SHRINK_DIR_PREFIX):
            continue
        try:
            n_removed = int(path.name.removeprefix(SHRINK_DIR_PREFIX))
        except ValueError:
            print(f"Warning: unexpected directory name '{path.name}', skipping")
            continue
        shrink_dirs.append((n_removed, path))

    yield from sorted(shrink_dirs, key=lambda item: item[0])


def collect_metrics(model_dir: Path) -> dict[str, list[float]]:
    """Collect performance metrics for a model directory."""
    series: dict[str, list[float]] = {"N_removed": []}
    for output_cfg in OUTPUT_CONFIGS.values():
        for metric_name in METRIC_NAMES:
            series[f"{output_cfg['metric_key']}_{metric_name}"] = []

    for n_removed, shrink_dir in iter_shrink_dirs(model_dir):
        metrics_path = shrink_dir / "performance_metrics.json"
        if not metrics_path.exists():
            print(f"Warning: missing metrics file in {shrink_dir}, skipping")
            continue

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        series["N_removed"].append(n_removed)
        for output_cfg in OUTPUT_CONFIGS.values():
            output_metrics = metrics.get(output_cfg["metric_key"], {})
            for metric_name in METRIC_NAMES:
                value = output_metrics.get(metric_name)
                series[f"{output_cfg['metric_key']}_{metric_name}"].append(value)

    return series


def compute_avg_ci_widths(model_dir: Path) -> dict[str, list[float]]:
    """Compute average confidence interval width for each output."""
    widths = {"N_removed": [], "TS": [], "TF": []}

    for n_removed, shrink_dir in iter_shrink_dirs(model_dir):
        prediction_path = shrink_dir / "Ypred.npy"
        if not prediction_path.exists():
            print(f"Warning: missing prediction array in {shrink_dir}, skipping")
            continue

        y_pred = np.load(prediction_path)
        if y_pred.ndim != 3 or y_pred.shape[1] < 2 or y_pred.shape[2] < 3:
            print(f"Warning: unexpected Ypred shape {y_pred.shape} in {shrink_dir}, skipping")
            continue

        widths["N_removed"].append(n_removed)
        for output_name, output_cfg in OUTPUT_CONFIGS.items():
            output_pred = y_pred[:, output_cfg["ypred_idx"], :]
            widths[output_name].append(np.mean(output_pred[:, 2] - output_pred[:, 1]))

    return widths


def plot_metric(ax: plt.Axes, model_series: dict[str, dict[str, list[float]]], title: str) -> None:
    """Plot a single metric axis for all models and outputs."""
    metric_spec = METRIC_SPECS[title]

    for model_label, config in MODEL_CONFIGS.items():
        series = model_series[model_label]
        if not series["N_removed"]:
            continue

        for output_name, output_cfg in OUTPUT_CONFIGS.items():
            ax.plot(
                series["N_removed"],
                series[f"{output_cfg['metric_key']}_{metric_spec['key']}"],
                color=config["colors"][output_name],
                marker=config["marker"],
                linestyle="-",
                alpha=0.7,
                label=f"{model_label} {output_name}",
            )

    ax.set_xlabel("Number of Sensors Removed", fontsize=12)
    ax.set_ylabel(metric_spec["ylabel"], fontsize=12)
    if "ylim" in metric_spec:
        ax.set_ylim(*metric_spec["ylim"])
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)


def plot_ci_widths(ax: plt.Axes, ci_width_series: dict[str, dict[str, list[float]]]) -> None:
    """Plot average CI width for all models and outputs."""
    for model_label, config in MODEL_CONFIGS.items():
        series = ci_width_series[model_label]
        if not series["N_removed"]:
            continue

        for output_name in OUTPUT_CONFIGS:
            ax.plot(
                series["N_removed"],
                series[output_name],
                color=config["colors"][output_name],
                marker=config["marker"],
                linestyle="-",
                alpha=0.7,
                label=f"{model_label} {output_name}",
            )

    ax.set_xlabel("Number of Sensors Removed", fontsize=12)
    ax.set_ylabel("Avg. CI Width (°C)", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    model_metric_series: dict[str, dict[str, list[float]]] = {}
    model_ci_width_series: dict[str, dict[str, list[float]]] = {}

    for model_label, config in MODEL_CONFIGS.items():
        model_dir = base_dir / config["directory"]
        try:
            model_metric_series[model_label] = collect_metrics(model_dir)
            model_ci_width_series[model_label] = compute_avg_ci_widths(model_dir)
        except FileNotFoundError as error:
            print(error)
            model_metric_series[model_label] = {"N_removed": []}
            model_ci_width_series[model_label] = {"N_removed": [], "TS": [], "TF": []}

    for model_label, series in model_metric_series.items():
        print(f"{model_label} metrics points: {len(series['N_removed'])}")

    fig = plt.figure(figsize=(12, 8))
    grid = gridspec.GridSpec(nrows=2, ncols=2, hspace=0.2, wspace=0.2)

    plot_metric(fig.add_subplot(grid[0, 0]), model_metric_series, "R²")
    plot_ci_widths(fig.add_subplot(grid[0, 1]), model_ci_width_series)
    plot_metric(fig.add_subplot(grid[1, 0]), model_metric_series, "MAE")
    plot_metric(fig.add_subplot(grid[1, 1]), model_metric_series, "RMSE")

    fig.tight_layout()
    output_path = base_dir / "HTTF_shrink_metrics.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
