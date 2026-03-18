from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch


_DEFAULT_PLOT_LABELS = ("Solar", "Wind")
_DEFAULT_PLOT_COLORS = (
    "orangered",
    "dodgerblue",
    "mediumseagreen",
    "mediumpurple",
    "goldenrod",
    "slategray",
)


def _extract_predictions(model_output):
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output



def _to_numpy_2d(values):
    array = np.asarray(values)
    if array.ndim == 1:
        return array[:, None]
    return array



def _inverse_transform_outputs(values, scaler_y=None):
    if scaler_y is None:
        return values

    values_2d = _to_numpy_2d(values)
    transformed = scaler_y.inverse_transform(values_2d)

    if np.asarray(values).ndim == 1:
        return transformed[:, 0]
    return transformed



def _resolve_output_labels(n_outputs, labels=None):
    if labels is None:
        base_labels = list(_DEFAULT_PLOT_LABELS)
    elif isinstance(labels, str):
        base_labels = [labels]
    else:
        base_labels = list(labels)

    if len(base_labels) < n_outputs:
        base_labels.extend(
            [f"Output {index + 1}" for index in range(len(base_labels), n_outputs)]
        )

    return base_labels[:n_outputs]



def _resolve_output_colors(n_outputs):
    return [
        _DEFAULT_PLOT_COLORS[index % len(_DEFAULT_PLOT_COLORS)]
        for index in range(n_outputs)
    ]



def predict_deterministic(model, data_loader, scaler_y=None, device=torch.device("cpu")):
    model.eval()
    all_predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            true_values.append(targets.cpu().numpy())
            outputs = _extract_predictions(model(inputs))
            all_predictions.append(outputs.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    true_values = np.concatenate(true_values, axis=0)

    if scaler_y is not None:
        true_values = _inverse_transform_outputs(true_values, scaler_y)
        all_predictions = _inverse_transform_outputs(all_predictions, scaler_y)

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
    """
    Run MC sampling through a model to estimate predictive uncertainty.
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_trues.append(targets.cpu().numpy())

            def sample_once():
                outputs = _extract_predictions(model(inputs))
                return outputs.detach().cpu().numpy()

            batch_preds = Parallel(n_jobs=n_jobs)(
                delayed(sample_once)() for _ in range(n_samples)
            )
            all_preds.append(np.stack(batch_preds, axis=0))

    all_preds = np.concatenate(all_preds, axis=1)
    true_vals = np.concatenate(all_trues, axis=0)

    if scaler_y is not None:
        true_vals = _inverse_transform_outputs(true_vals, scaler_y)
        all_preds = np.array([
            _inverse_transform_outputs(sample_predictions, scaler_y)
            for sample_predictions in all_preds
        ])

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
):
    """
    Plot deterministic or variational predictions for static and rolling workflows.

    Args:
        preds: Predicted values with shape (T,) or (T, K).
        trues: Ground-truth values with shape matching ``preds``.
        title: Optional plot title prefix applied per output.
        labels: Optional output labels. Defaults to Solar/Wind and falls back to
            generic output names when needed.
        n_display: Optional maximum number of timesteps to display. If ``None``,
            show the full series.
        lower: Optional lower uncertainty bound with shape matching ``preds``.
        upper: Optional upper uncertainty bound with shape matching ``preds``.
    """
    preds = _to_numpy_2d(preds)
    trues = _to_numpy_2d(trues)

    if preds.shape != trues.shape:
        raise ValueError(
            f"preds and trues must have matching shapes, got {preds.shape} and {trues.shape}."
        )

    lower = None if lower is None else _to_numpy_2d(lower)
    upper = None if upper is None else _to_numpy_2d(upper)

    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must both be provided when plotting uncertainty bands.")

    if lower is not None and (lower.shape != preds.shape or upper.shape != preds.shape):
        raise ValueError(
            "lower and upper must match preds/trues shape when plotting uncertainty bands."
        )

    total_points = preds.shape[0]
    display_points = total_points if n_display is None else min(total_points, int(n_display))
    x = np.arange(display_points)
    output_labels = _resolve_output_labels(preds.shape[1], labels)
    output_colors = _resolve_output_colors(preds.shape[1])
    band_label = "95% CI"

    for output_index, output_label in enumerate(output_labels):
        color = output_colors[output_index]
        plt.figure(figsize=(16, 6))
        plt.plot(x, trues[:display_points, output_index], "k", label="Actual")
        plt.plot(
            x,
            preds[:display_points, output_index],
            color=color,
            linestyle="--",
            label="Predicted",
        )

        if lower is not None and upper is not None:
            plt.fill_between(
                x,
                lower[:display_points, output_index],
                upper[:display_points, output_index],
                alpha=0.3,
                color=color,
                label=band_label,
            )

        plt.xlabel("Time (minutes)")
        plt.ylabel(output_label)
        plt.grid()
        if title:
            plt.title(f"{title} — {output_label}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()



def calculate_and_display_metrics(true_values, mean_predictions):
    """
    Calculate and print R², MAE, and RMSE for each output.
    """
    true_values = _to_numpy_2d(true_values)
    mean_predictions = _to_numpy_2d(mean_predictions)

    if true_values.shape != mean_predictions.shape:
        raise ValueError(
            "true_values and mean_predictions must have matching shapes to compute metrics."
        )

    n_outputs = true_values.shape[1]

    r2_scores = [r2_score(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    mae_scores = [mean_absolute_error(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    rmse_scores = [np.sqrt(mean_squared_error(true_values[:, i], mean_predictions[:, i])) for i in range(n_outputs)]

    for i in range(n_outputs):
        print(f"Output {i+1}:")
        print(f"  R² score: {r2_scores[i]:.4f}")
        print(f"  MAE score: {mae_scores[i]:.4f}")
        print(f"  RMSE score: {rmse_scores[i]:.4f}")
        print()
