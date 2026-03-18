from __future__ import annotations

from typing import Sequence

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch

DEFAULT_LABELS = ("Solar", "Wind")
DEFAULT_ACTUAL_LABEL = "Actual"
DEFAULT_PREDICTED_LABEL = "Predicted"
DEFAULT_UNCERTAINTY_LABEL = "Uncertainty band"
DEFAULT_FIGSIZE = (16, 6)



def _extract_predictions(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output



def _ensure_2d(array_like):
    array = np.asarray(array_like)
    if array.ndim == 1:
        return array[:, np.newaxis]
    return array



def _resolve_labels(labels: Sequence[str] | None, n_outputs: int) -> list[str]:
    if labels is None:
        labels = DEFAULT_LABELS

    resolved = list(labels[:n_outputs])
    if len(resolved) < n_outputs:
        resolved.extend([f"Output {index + 1}" for index in range(len(resolved), n_outputs)])
    return resolved



def predict_deterministic(model, data_loader, scaler_y=None, device=torch.device("cpu")):
    model.eval()
    all_predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            true_values.append(targets.detach().cpu().numpy())
            outputs = _extract_predictions(model(inputs))
            all_predictions.append(outputs.detach().cpu().numpy())

    all_predictions = _ensure_2d(np.concatenate(all_predictions, axis=0))
    true_values = _ensure_2d(np.concatenate(true_values, axis=0))

    if scaler_y is not None:
        true_values = scaler_y.inverse_transform(true_values)
        all_predictions = scaler_y.inverse_transform(all_predictions)

    return all_predictions, true_values



def predict_with_uncertainty(
    model,
    test_loader,
    n_samples=100,
    scaler_y=None,
    device=torch.device("cpu"),
    n_jobs=4,
    alpha=0.05,
):
    """Run Monte Carlo sampling through a model to estimate predictive uncertainty."""
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_trues.append(targets.detach().cpu().numpy())

            def sample_once():
                outputs = _extract_predictions(model(inputs))
                return outputs.detach().cpu().numpy()

            batch_preds = Parallel(n_jobs=n_jobs)(
                delayed(sample_once)() for _ in range(n_samples)
            )
            all_preds.append(np.stack([_ensure_2d(prediction) for prediction in batch_preds], axis=0))

    all_preds = np.concatenate(all_preds, axis=1)
    true_vals = _ensure_2d(np.concatenate(all_trues, axis=0))

    if scaler_y is not None:
        true_vals = scaler_y.inverse_transform(true_vals)
        all_preds = np.array([scaler_y.inverse_transform(prediction) for prediction in all_preds])

    mean_preds = np.mean(all_preds, axis=0)
    lower = np.percentile(all_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)

    return mean_preds, true_vals, lower, upper



def plot_predictions(
    preds,
    trues,
    title=None,
    labels=None,
    n_display=None,
    lower=None,
    upper=None,
    actual_label=DEFAULT_ACTUAL_LABEL,
    predicted_label=DEFAULT_PREDICTED_LABEL,
    uncertainty_label=DEFAULT_UNCERTAINTY_LABEL,
    x_label="Time (minutes)",
    figsize=DEFAULT_FIGSIZE,
):
    """Plot deterministic or variational predictions for static or rolling evaluation runs."""
    preds = _ensure_2d(preds)
    trues = _ensure_2d(trues)

    if preds.shape != trues.shape:
        raise ValueError(
            f"preds and trues must have matching shapes, got {preds.shape} and {trues.shape}."
        )

    lower_array = None if lower is None else _ensure_2d(lower)
    upper_array = None if upper is None else _ensure_2d(upper)
    if (lower_array is None) != (upper_array is None):
        raise ValueError("lower and upper must both be provided when plotting uncertainty bands.")
    if lower_array is not None and (lower_array.shape != preds.shape or upper_array.shape != preds.shape):
        raise ValueError("lower and upper must match preds/trues shape when plotting uncertainty bands.")

    n_outputs = preds.shape[1]
    resolved_labels = _resolve_labels(labels, n_outputs)
    n_points = preds.shape[0] if n_display is None else min(preds.shape[0], n_display)
    x = np.arange(n_points)

    for output_index, output_label in enumerate(resolved_labels):
        color = "orangered" if output_index == 0 else "dodgerblue"
        plt.figure(figsize=figsize)
        plt.plot(x, trues[:n_points, output_index], "k", label=actual_label)
        plt.plot(
            x,
            preds[:n_points, output_index],
            color=color,
            linestyle="--",
            label=predicted_label,
        )

        if lower_array is not None and upper_array is not None:
            plt.fill_between(
                x,
                lower_array[:n_points, output_index],
                upper_array[:n_points, output_index],
                alpha=0.3,
                color=color,
                label=uncertainty_label,
            )

        plt.xlabel(x_label)
        plt.ylabel(output_label)
        plt.grid()
        if title:
            plt.title(f"{title} — {output_label}")
        else:
            plt.title(output_label)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()



def calculate_and_display_metrics(true_values, mean_predictions):
    """Calculate and print R², MAE, and RMSE for each output."""
    true_values = _ensure_2d(true_values)
    mean_predictions = _ensure_2d(mean_predictions)

    if true_values.shape != mean_predictions.shape:
        raise ValueError(
            "true_values and mean_predictions must have matching shapes, "
            f"got {true_values.shape} and {mean_predictions.shape}."
        )

    n_outputs = true_values.shape[1]

    r2_scores = [r2_score(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    mae_scores = [mean_absolute_error(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    rmse_scores = [np.sqrt(mean_squared_error(true_values[:, i], mean_predictions[:, i])) for i in range(n_outputs)]

    for i in range(n_outputs):
        print(f"Output {i + 1}:")
        print(f"  R² score: {r2_scores[i]:.4f}")
        print(f"  MAE score: {mae_scores[i]:.4f}")
        print(f"  RMSE score: {rmse_scores[i]:.4f}")
        print()


__all__ = [
    "calculate_and_display_metrics",
    "plot_predictions",
    "predict_deterministic",
    "predict_with_uncertainty",
]
