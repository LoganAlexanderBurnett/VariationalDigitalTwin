from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

BASE_DIR = Path(__file__).resolve().parent
TRUE_VALUES_PATH = BASE_DIR / "variational" / "GRU" / "Ytrue.npy"
VERTICAL_LINE_EVERY = 5485
X_LIMIT = 340070

PLOT_CONFIGS = (
    {
        "title": "GRU vs VGRU - TS",
        "deterministic_label": "GRU",
        "variational_label": "VGRU Mean",
        "deterministic_path": BASE_DIR / "deterministic" / "GRU" / "Ypred_rescaled.npy",
        "variational_path": BASE_DIR / "variational" / "GRU" / "Ypred.npy",
        "output_index": 0,
        "ylabel": "Solid Temperature (°C)",
        "zoom_start": 191975,
        "zoom_width": 5473,
        "inset_anchor": (0.52, 0.65, 0.1, 0.1),
        "deterministic_color": "b",
        "variational_color": "darkcyan",
        "output_name": "compare_gru_vgru_ts.png",
    },
    {
        "title": "GRU vs VGRU - TF",
        "deterministic_label": "GRU",
        "variational_label": "VGRU Mean",
        "deterministic_path": BASE_DIR / "deterministic" / "GRU" / "Ypred_rescaled.npy",
        "variational_path": BASE_DIR / "variational" / "GRU" / "Ypred.npy",
        "output_index": 1,
        "ylabel": "Fluid Temperature (°C)",
        "zoom_start": 87760,
        "zoom_width": 5473,
        "inset_anchor": (0.34, 0.75, 0.1, 0.1),
        "deterministic_color": "b",
        "variational_color": "darkcyan",
        "output_name": "compare_gru_vgru_tf.png",
    },
    {
        "title": "LSTM vs VLSTM - TS",
        "deterministic_label": "LSTM",
        "variational_label": "VLSTM Mean",
        "deterministic_path": BASE_DIR / "deterministic" / "LSTM" / "Ypred_rescaled.npy",
        "variational_path": BASE_DIR / "variational" / "LSTM" / "Ypred.npy",
        "output_index": 0,
        "ylabel": "Solid Temperature (°C)",
        "zoom_start": 191975,
        "zoom_width": 5473,
        "inset_anchor": (0.52, 0.65, 0.1, 0.1),
        "deterministic_color": "firebrick",
        "variational_color": "darkorange",
        "output_name": "compare_lstm_vlstm_ts.png",
    },
    {
        "title": "LSTM vs VLSTM - TF",
        "deterministic_label": "LSTM",
        "variational_label": "VLSTM Mean",
        "deterministic_path": BASE_DIR / "deterministic" / "LSTM" / "Ypred_rescaled.npy",
        "variational_path": BASE_DIR / "variational" / "LSTM" / "Ypred.npy",
        "output_index": 1,
        "ylabel": "Fluid Temperature (°C)",
        "zoom_start": 87760,
        "zoom_width": 5473,
        "inset_anchor": (0.34, 0.75, 0.1, 0.1),
        "deterministic_color": "firebrick",
        "variational_color": "darkorange",
        "output_name": "compare_lstm_vlstm_tf.png",
    },
)


def load_deterministic_predictions(path: Path) -> np.ndarray:
    predictions = np.load(path)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"Expected deterministic predictions with shape (N, 2), got {predictions.shape} from {path}")
    return predictions


def load_variational_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.load(path)
    if predictions.ndim != 3 or predictions.shape[1] != 2 or predictions.shape[2] != 3:
        raise ValueError(
            f"Expected variational predictions with shape (N, 2, 3), got {predictions.shape} from {path}"
        )
    mean_predictions = predictions[:, :, 0]
    lower_ci = predictions[:, :, 1]
    upper_ci = predictions[:, :, 2]
    return mean_predictions, lower_ci, upper_ci


def add_vertical_lines(ax: plt.Axes, max_x: int) -> None:
    for x_val in np.arange(VERTICAL_LINE_EVERY, max_x, VERTICAL_LINE_EVERY):
        ax.axvline(x=x_val, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)


def plot_comparison(
    true_values: np.ndarray,
    deterministic_predictions: np.ndarray,
    variational_mean: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    *,
    title: str,
    deterministic_label: str,
    variational_label: str,
    output_index: int,
    ylabel: str,
    zoom_start: int,
    zoom_width: int,
    inset_anchor: tuple[float, float, float, float],
    deterministic_color: str,
    variational_color: str,
    output_path: Path,
) -> None:
    n_points = min(
        len(true_values),
        len(deterministic_predictions),
        len(variational_mean),
        len(lower_ci),
        len(upper_ci),
    )
    x = np.arange(n_points)
    zoom_start = min(zoom_start, n_points - 1)
    zoom_end = min(zoom_start + zoom_width, n_points - 1)
    mask = (x >= zoom_start) & (x <= zoom_end)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)
    ax.plot(x, true_values[:n_points, output_index], label="Actual", color="k", linewidth=2)
    ax.plot(
        x,
        deterministic_predictions[:n_points, output_index],
        label=deterministic_label,
        color=deterministic_color,
        linestyle="--",
    )
    ax.plot(
        x,
        variational_mean[:n_points, output_index],
        label=variational_label,
        color=variational_color,
        linestyle="--",
    )
    ax.fill_between(
        x,
        lower_ci[:n_points, output_index],
        upper_ci[:n_points, output_index],
        alpha=0.3,
        color=variational_color,
        label="95% CI",
    )
    add_vertical_lines(ax, n_points)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps (30 seconds/step)")
    ax.set_xlim(0, min(X_LIMIT, n_points))
    legend = ax.legend(loc="upper left")

    axins = inset_axes(
        ax,
        width=1.75,
        height=1.75,
        bbox_to_anchor=inset_anchor,
        bbox_transform=ax.transAxes,
        loc="center",
    )
    axins.plot(x[mask], true_values[:n_points, output_index][mask], color="k", linewidth=2)
    axins.plot(
        x[mask],
        deterministic_predictions[:n_points, output_index][mask],
        color=deterministic_color,
        linestyle="--",
    )
    axins.plot(
        x[mask],
        variational_mean[:n_points, output_index][mask],
        color=variational_color,
        linestyle="--",
        linewidth=2,
    )
    axins.fill_between(
        x[mask],
        lower_ci[:n_points, output_index][mask],
        upper_ci[:n_points, output_index][mask],
        color=variational_color,
        alpha=0.3,
    )
    axins.set_xlim(zoom_start, zoom_end)

    fig.savefig(output_path, dpi=300, bbox_inches="tight", bbox_extra_artists=[axins, legend])
    plt.close(fig)


def main() -> None:
    true_values = np.load(TRUE_VALUES_PATH)
    if true_values.ndim != 2 or true_values.shape[1] != 2:
        raise ValueError(f"Expected true values with shape (N, 2), got {true_values.shape} from {TRUE_VALUES_PATH}")

    for config in PLOT_CONFIGS:
        deterministic_predictions = load_deterministic_predictions(config["deterministic_path"])
        variational_mean, lower_ci, upper_ci = load_variational_predictions(config["variational_path"])
        output_path = BASE_DIR / config["output_name"]

        plot_comparison(
            true_values,
            deterministic_predictions,
            variational_mean,
            lower_ci,
            upper_ci,
            title=config["title"],
            deterministic_label=config["deterministic_label"],
            variational_label=config["variational_label"],
            output_index=config["output_index"],
            ylabel=config["ylabel"],
            zoom_start=config["zoom_start"],
            zoom_width=config["zoom_width"],
            inset_anchor=config["inset_anchor"],
            deterministic_color=config["deterministic_color"],
            variational_color=config["variational_color"],
            output_path=output_path,
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
