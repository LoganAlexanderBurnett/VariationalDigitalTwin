import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from torchinfo import summary
import json
import os
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf import (
    DeterministicLSTMModel,
    build_tensor_dataloader,
    evaluate_deterministic_model,
    plot_loss_curves,
    plot_prediction_series,
    prepare_csv_autoregressive_splits,
    print_metrics_report,
    print_split_shapes,
)
from httf.trainer import set_random_seed, train_deterministic


# parse command-line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-dir", "-d",
    required=True,
    help="Path to a shrink_train_## folder containing train.csv, valid.csv, test.csv"
)
args = parser.parse_args()

# now change into that directory
os.chdir(args.data_dir)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_random_seed()

prepared = prepare_csv_autoregressive_splits(
    ".",
    train_name="train.csv",
    test_name="test.csv",
    valid_name="valid.csv",
    lookback=10,
    device=device,
    interpolate_columns=["TS"],
)
scaler = prepared["scaler"]
Xtrain_tensor, Ytrain_tensor = prepared["train"]
Xtest_tensor, Ytest_tensor = prepared["test"]
Xvalid_tensor, Yvalid_tensor = prepared["valid"]
print_split_shapes(prepared["train"], prepared["test"], prepared["valid"])


# Instantiate the model
batch_size = 512
model = DeterministicLSTMModel(input_size=2, hidden_size1=96, hidden_size2=16, hidden_size3=48, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000023)

# Training loop
epochs = 50
train_loader = build_tensor_dataloader(Xtrain_tensor, Ytrain_tensor, batch_size=batch_size)
valid_loader = build_tensor_dataloader(Xvalid_tensor, Yvalid_tensor, batch_size=batch_size)

start = time.time()

history = train_deterministic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    num_epochs=epochs,
    device=device,
    val_loader=valid_loader,
)

end = time.time()
print(f"Training took {end - start}s")

plot_loss_curves(history, "training.png")

test_loader = build_tensor_dataloader(Xtest_tensor, Ytest_tensor, batch_size=batch_size)
evaluation = evaluate_deterministic_model(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
    output_dir=".",
    output_names=["output_1", "output_2"],
)
print_metrics_report(evaluation["metrics"])

plot_prediction_series(
    evaluation["true_values"][:, 0],
    evaluation["predictions"][:, 0],
    "testing_ts.png",
    ylabel="Solid Temperature (°C)",
    predicted_label="LSTM",
    color="red",
    test_length=prepared["test_length"],
)
plot_prediction_series(
    evaluation["true_values"][:, 1],
    evaluation["predictions"][:, 1],
    "testing_tf.png",
    ylabel="Fluid Temperature (°C)",
    predicted_label="LSTM",
    color="red",
    test_length=prepared["test_length"],
)

