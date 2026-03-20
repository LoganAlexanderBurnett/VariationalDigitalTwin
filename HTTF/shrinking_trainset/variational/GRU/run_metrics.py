import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf import (
    VariationalGRUModel,
    build_tensor_dataloader,
    evaluate_variational_model,
    plot_loss_curves,
    plot_prediction_series,
    prepare_csv_autoregressive_splits,
    print_metrics_report,
    print_split_shapes,
)
from httf.trainer import set_random_seed, train_variational


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

plot_loss_curves(history, "training.png")

start = time.time()
evaluation = evaluate_variational_model(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
    output_dir=".",
    n_samples=100,
    output_names=["output_1", "output_2"],
)
end = time.time()
print(f"Prediction time: {end - start:.4f} seconds")
print_metrics_report(evaluation["metrics"])

plot_prediction_series(
    evaluation["true_values"][:, 0],
    evaluation["mean_predictions"][:, 0],
    "testing_ts.png",
    ylabel="Solid Temperature (°C)",
    predicted_label="Mean Prediction",
    lower=evaluation["lower_ci"][:, 0],
    upper=evaluation["upper_ci"][:, 0],
    color="red",
    test_length=prepared["test_length"],
)
plot_prediction_series(
    evaluation["true_values"][:, 1],
    evaluation["mean_predictions"][:, 1],
    "testing_tf.png",
    ylabel="Fluid Temperature (°C)",
    predicted_label="Mean Prediction",
    lower=evaluation["lower_ci"][:, 1],
    upper=evaluation["upper_ci"][:, 1],
    color="red",
    test_length=prepared["test_length"],
)
