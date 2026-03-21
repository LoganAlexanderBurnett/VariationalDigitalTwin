from pathlib import Path
import sys
import time

import torch
import torch.nn as nn

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "httf").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf import (
    DeterministicGRUModel,
    build_tensor_dataloader,
    evaluate_deterministic_model,
    plot_loss_curves,
    plot_prediction_series,
    prepare_csv_autoregressive_splits,
    print_metrics_report,
    print_split_shapes,
    set_random_seed,
)
from httf.trainer import train_deterministic

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

batch_size = 512
timesteps = Xtrain_tensor.shape[1]
model = DeterministicGRUModel(input_size=2, hidden_size1=96, hidden_size2=16, hidden_size3=48, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000023)

train_loader = build_tensor_dataloader(Xtrain_tensor, Ytrain_tensor, batch_size=batch_size, shuffle=False)
valid_loader = build_tensor_dataloader(Xvalid_tensor, Yvalid_tensor, batch_size=batch_size, shuffle=False)
test_loader = build_tensor_dataloader(Xtest_tensor, Ytest_tensor, batch_size=batch_size, shuffle=False)

start = time.time()
history = train_deterministic(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    loss_fn=criterion,
    num_epochs=50,
    device=device,
    val_loader=valid_loader,
)
end = time.time()
print(f"Training took {end - start}s")

plot_loss_curves(history, DATA_DIR / "training.png")

evaluation = evaluate_deterministic_model(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
    output_dir=DATA_DIR,
    output_names=["output_1", "output_2"],
)
print_metrics_report(evaluation["metrics"])

plot_prediction_series(
    evaluation["true_values"][:, 0],
    evaluation["predictions"][:, 0],
    DATA_DIR / "DETERM_TS.png",
    ylabel="Solid Temperature (°C)",
    predicted_label="GRU",
    color="red",
    test_length=prepared["test_length"],
)
plot_prediction_series(
    evaluation["true_values"][:, 1],
    evaluation["predictions"][:, 1],
    DATA_DIR / "DETERM_TF.png",
    ylabel="Fluid Temperature (°C)",
    predicted_label="GRU",
    color="red",
    test_length=prepared["test_length"],
)
