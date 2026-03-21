from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torchinfo import summary

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "httf").exists())
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
    set_random_seed,
)
from httf.trainer import train_variational

DATA_DIR = Path(__file__).resolve().parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_random_seed()

prepared = prepare_csv_autoregressive_splits(
    DATA_DIR,
    train_name="TF_TS_train.csv",
    test_name="TF_TS_test.csv",
    valid_name="TF_TS_valid.csv",
    lookback=10,
    device=device,
    interpolate_columns=["TS"],
)
print_split_shapes(prepared["train"], prepared["test"], prepared["valid"])

scaler = prepared["scaler"]
Xtrain_tensor, Ytrain_tensor = prepared["train"]
Xtest_tensor, Ytest_tensor = prepared["test"]
Xvalid_tensor, Yvalid_tensor = prepared["valid"]

batch_size = 256
timesteps = Xtrain_tensor.shape[1]
model = VariationalGRUModel(in_features=2, hidden_size1=48, hidden_size2=64, hidden_size3=32, out_features=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00018)

summary(model, input_size=(batch_size, timesteps, 2), device=device)

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

plot_loss_curves(history, DATA_DIR / "training.png", yscale="log")

evaluation = evaluate_variational_model(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
    output_dir=DATA_DIR,
    n_samples=250,
    output_names=["output_1", "output_2"],
)
print_metrics_report(evaluation["metrics"])

plot_prediction_series(
    evaluation["true_values"][:, 0],
    evaluation["mean_predictions"][:, 0],
    DATA_DIR / "VBBL_TS.png",
    ylabel="Solid Temperature (°C)",
    predicted_label="Variational GRU Mean",
    lower=evaluation["lower_ci"][:, 0],
    upper=evaluation["upper_ci"][:, 0],
    color="darkcyan",
    test_length=prepared["test_length"],
)
plot_prediction_series(
    evaluation["true_values"][:, 1],
    evaluation["mean_predictions"][:, 1],
    DATA_DIR / "VBBL_TF.png",
    ylabel="Fluid Temperature (°C)",
    predicted_label="Variational GRU Mean",
    lower=evaluation["lower_ci"][:, 1],
    upper=evaluation["upper_ci"][:, 1],
    color="darkcyan",
    test_length=prepared["test_length"],
)
