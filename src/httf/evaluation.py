import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_handler import prepare_autoregressive_splits, report_nan_rows
from .predict import predict_deterministic, predict_with_uncertainty


def load_csv_splits(data_dir, train_name, test_name, valid_name):
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / train_name)
    test = pd.read_csv(data_dir / test_name)
    valid = pd.read_csv(data_dir / valid_name)
    return train, test, valid


def prepare_csv_autoregressive_splits(
    data_dir,
    train_name,
    test_name,
    valid_name,
    lookback,
    device=torch.device("cpu"),
    nan_column="TS",
    interpolate_columns=None,
):
    train, test, valid = load_csv_splits(data_dir, train_name, test_name, valid_name)
    report_nan_rows(train, nan_column, "Train")
    report_nan_rows(test, nan_column, "Test")
    report_nan_rows(valid, nan_column, "Validation")

    prepared = prepare_autoregressive_splits(
        train,
        test,
        valid,
        lookback=lookback,
        device=device,
        interpolate_columns=interpolate_columns,
    )

    return {
        "train_df": train,
        "test_df": test,
        "valid_df": valid,
        "test_length": len(test),
        **prepared,
    }


def print_split_shapes(train_tensors, test_tensors, valid_tensors):
    Xtrain_tensor, Ytrain_tensor = train_tensors
    Xtest_tensor, Ytest_tensor = test_tensors
    Xvalid_tensor, Yvalid_tensor = valid_tensors
    print(
        f"""Xtrain, Ytrain: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xtest,   Ytest: {Xtest_tensor.shape}, {Ytest_tensor.shape}
Xvalid, Yvalid: {Xvalid_tensor.shape}, {Yvalid_tensor.shape}"""
    )


def plot_loss_curves(history, output_path, title="Loss Curve", ylabel="Loss (MSE)", yscale=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_losses"], label="Training Loss")
    if history.get("val_losses"):
        plt.plot(history["val_losses"], label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    if yscale:
        plt.yscale(yscale)
    plt.legend()
    plt.grid()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_prediction_series(
    y_true,
    y_pred,
    output_path,
    ylabel,
    predicted_label,
    lower=None,
    upper=None,
    color="red",
    test_length=None,
    ylim=(0, 1200),
    vertical_line_every=5485,
):
    plt.figure(figsize=(16, 6))
    x = np.arange(len(y_true))

    for x_val in np.arange(vertical_line_every, len(y_true), vertical_line_every):
        plt.vlines(x=x_val, ymin=-100, ymax=1500, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.plot(x, y_true, "b", label="Actual")
    plt.plot(x, y_pred, color=color, linestyle="--", label=predicted_label)

    if lower is not None and upper is not None:
        plt.fill_between(x, lower, upper, color=color, alpha=0.2, label="95% CI")

    plt.xlabel("Time Steps (30 seconds/step)")
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlim(0, len(y_true) if test_length is None else test_length)
    plt.legend(loc="upper left")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def calculate_regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
    return {
        "r2": float(r2),
        "mae": float(mae),
        "mape": float(mape),
        "rmse": float(rmse),
        "rmspe": float(rmspe),
    }


def build_metrics_report(y_true, y_pred, output_names=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    output_names = output_names or [f"output_{index + 1}" for index in range(y_true.shape[1])]
    metrics = {}
    for index, output_name in enumerate(output_names):
        metrics[output_name] = calculate_regression_metrics(y_true[:, index], y_pred[:, index])
    return metrics


def print_metrics_report(metrics):
    for output_name, output_metrics in metrics.items():
        print(f"--- Metrics for {output_name} ---")
        print(f"R^2 Score: {output_metrics['r2']}")
        print(f"MAE:        {output_metrics['mae']}")
        print(f"MAPE:       {output_metrics['mape']}%")
        print(f"RMSE:       {output_metrics['rmse']}")
        print(f"RMSPE:      {output_metrics['rmspe']}%\n")


def save_metrics_json(metrics, output_path):
    with open(output_path, "w") as fp:
        json.dump(metrics, fp, indent=4)


def evaluate_deterministic_model(
    model,
    test_loader,
    scaler_y,
    device=torch.device("cpu"),
    output_dir=".",
    output_names=None,
):
    output_dir = Path(output_dir)
    predictions, true_values = predict_deterministic(model, test_loader, scaler_y=scaler_y, device=device)
    np.save(output_dir / "Ypred_rescaled.npy", predictions)
    np.save(output_dir / "Ytrue_rescaled.npy", true_values)
    metrics = build_metrics_report(true_values, predictions, output_names=output_names)
    save_metrics_json(metrics, output_dir / "performance_metrics.json")
    return {
        "predictions": predictions,
        "true_values": true_values,
        "metrics": metrics,
    }


def evaluate_variational_model(
    model,
    test_loader,
    scaler_y,
    device=torch.device("cpu"),
    output_dir=".",
    n_samples=100,
    output_names=None,
):
    output_dir = Path(output_dir)
    mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(
        model,
        test_loader,
        n_samples=n_samples,
        scaler_y=scaler_y,
        device=device,
    )
    stacked_predictions = np.stack((mean_predictions, lower_ci, upper_ci), axis=2)
    np.save(output_dir / "Ypred.npy", stacked_predictions)
    np.save(output_dir / "Ytrue.npy", true_values)
    metrics = build_metrics_report(true_values, mean_predictions, output_names=output_names)
    save_metrics_json(metrics, output_dir / "performance_metrics.json")
    return {
        "mean_predictions": mean_predictions,
        "true_values": true_values,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "metrics": metrics,
    }
