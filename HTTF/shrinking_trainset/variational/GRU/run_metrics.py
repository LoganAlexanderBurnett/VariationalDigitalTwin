import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf.data_handler import build_tensor_dataloader, prepare_autoregressive_splits, report_nan_rows
from httf.models import VariationalGRUModel
from httf.predict import predict_with_uncertainty
from httf.trainer import set_random_seed, train_variational


def plot_with_ci(y_true, y_pred_mean, y_pred_lower, y_pred_upper, ylabel, fname, test_length):
    plt.figure(figsize=(16, 6))

    for x_val in np.arange(5485, len(y_true), 5485):
        plt.vlines(x=x_val, ymin=-100, ymax=1500, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    x = np.arange(len(y_true))
    plt.plot(x, y_true, "b", label="Actual")
    plt.plot(x, y_pred_mean, "r", linestyle="--", label="Mean Prediction")
    plt.fill_between(x, y_pred_lower, y_pred_upper, color="r", alpha=0.2, label="95% CI")

    plt.xlabel("Time Steps (30 seconds/step)")
    plt.ylabel(ylabel)
    plt.ylim(0, 1200)
    plt.xlim(0, test_length)
    plt.legend(loc="upper left")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2)) * 100
    return r2, mae, mape, rmse, rmspe


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir",
    "-d",
    required=True,
    help="Path to a shrink_train_## folder containing train.csv, valid.csv, test.csv",
)
args = parser.parse_args()

os.chdir(args.data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_random_seed()

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
valid = pd.read_csv("valid.csv")

test_length = len(test)

report_nan_rows(train, "TS", "Train")
report_nan_rows(test, "TS", "Test")
report_nan_rows(valid, "TS", "Validation")

prepared = prepare_autoregressive_splits(
    train,
    test,
    valid,
    lookback=10,
    device=device,
    interpolate_columns=["TS"],
)
scaler = prepared["scaler"]
Xtrain_tensor, Ytrain_tensor = prepared["train"]
Xtest_tensor, Ytest_tensor = prepared["test"]
Xvalid_tensor, Yvalid_tensor = prepared["valid"]

print(
    f"""Xtrain, Ytrain: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xtest,   Ytest: {Xtest_tensor.shape}, {Ytest_tensor.shape}
Xvalid, Yvalid: {Xvalid_tensor.shape}, {Yvalid_tensor.shape}"""
)

batch_size = 256
criterion = nn.MSELoss()
model = VariationalGRUModel(
    in_features=2,
    hidden_size1=48,
    hidden_size2=64,
    hidden_size3=32,
    out_features=2,
).to(device)
for gru in (model.gru1, model.gru2, model.gru3):
    gru.flatten_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=0.00018)

train_loader = build_tensor_dataloader(Xtrain_tensor, Ytrain_tensor, batch_size=batch_size)
valid_loader = build_tensor_dataloader(Xvalid_tensor, Yvalid_tensor, batch_size=batch_size)
test_loader = build_tensor_dataloader(Xtest_tensor, Ytest_tensor, batch_size=batch_size)

start = time.time()
history = train_variational(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    reconstruction_loss_fn=criterion,
    num_epochs=50,
    device=device,
    val_loader=valid_loader,
)
end = time.time()
print(f"Training took {end - start}s")

plt.figure(figsize=(10, 6))
plt.plot(history["train_losses"], label="Training Loss")
plt.plot(history["val_losses"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.savefig("training.png", bbox_inches="tight")
plt.close()

start = time.time()
mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(
    model,
    test_loader,
    n_samples=100,
    scaler_y=scaler,
    device=device,
)
end = time.time()
print(f"Prediction time: {end - start:.4f} seconds")

Ypred = np.stack((mean_predictions, lower_ci, upper_ci), axis=2)
np.save("Ypred.npy", Ypred)
np.save("Ytrue.npy", true_values)

print("Saved Ypred.npy with shape:", Ypred.shape)
print("Saved Ytrue.npy with shape:", true_values.shape)

Ypred_mean_1, Ypred_mean_2 = mean_predictions[:, 0], mean_predictions[:, 1]
Ypred_upper_1, Ypred_upper_2 = upper_ci[:, 0], upper_ci[:, 1]
Ypred_lower_1, Ypred_lower_2 = lower_ci[:, 0], lower_ci[:, 1]
Ytrue_1, Ytrue_2 = true_values[:, 0], true_values[:, 1]

plot_with_ci(Ytrue_1, Ypred_mean_1, Ypred_lower_1, Ypred_upper_1, "Solid Temperature (°C)", "testing_ts.png", test_length)
plot_with_ci(Ytrue_2, Ypred_mean_2, Ypred_lower_2, Ypred_upper_2, "Fluid Temperature (°C)", "testing_tf.png", test_length)

metrics = {}
for idx, (y_true, y_pred) in enumerate(((Ytrue_1, Ypred_mean_1), (Ytrue_2, Ypred_mean_2)), start=1):
    r2, mae, mape, rmse, rmspe = calculate_metrics(y_true, y_pred)

    print(f"--- Metrics for output #{idx} ---")
    print(f"R^2 Score: {r2}")
    print(f"MAE:        {mae}")
    print(f"MAPE:       {mape}%")
    print(f"RMSE:       {rmse}")
    print(f"RMSPE:      {rmspe}%\n")

    metrics[f"output_{idx}"] = {
        "r2": float(r2),
        "mae": float(mae),
        "mape": float(mape),
        "rmse": float(rmse),
        "rmspe": float(rmspe),
    }

with open("performance_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=4)

print("Saved all metrics to performance_metrics.json")
