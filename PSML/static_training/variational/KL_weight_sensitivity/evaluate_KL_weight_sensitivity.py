from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader

from psml.data_handler import *
from psml.models import GRUReparameterizationModel
from psml.predict import predict_with_uncertainty
from psml.trainer import set_random_seed


set_random_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def train_model_with_fixed_kl_weight(
    model,
    train_loader,
    val_loader,
    num_epochs,
    reconstruction_loss_fn,
    optimizer,
    kl_weight,
    device=torch.device("cpu"),
):
    model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()
            running_train_loss += total_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs, val_kl_loss = model(val_inputs)
                val_reconstruction_loss = reconstruction_loss_fn(val_outputs, val_targets)
                running_val_loss += (val_reconstruction_loss + kl_weight * val_kl_loss).item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, KL Weight: {kl_weight}"
        )

    return train_losses, val_losses


def save_arrays_to_csv(
    true_values: np.ndarray,
    mean_predictions: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    filename: Path,
):
    df = pd.DataFrame({
        'True Solar': true_values[:, 0],
        'True Wind': true_values[:, 1],
        'Predicted Mean Solar': mean_predictions[:, 0],
        'Predicted Mean Wind': mean_predictions[:, 1],
        'Lower CI Solar': lower_ci[:, 0],
        'Lower CI Wind': lower_ci[:, 1],
        'Upper CI Solar': upper_ci[:, 0],
        'Upper CI Wind': upper_ci[:, 1],
    })
    df.to_csv(filename, index=False)


def save_loss_plot(train_losses, val_losses, output_path: Path, title: str):
    plt.figure(figsize=(5, 4))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_plots(mean_predictions, true_values, lower_ci, upper_ci, output_dir: Path, n_display: int):
    labels = ["Solar", "Wind"]
    for idx, label in enumerate(labels):
        x = np.arange(min(n_display, mean_predictions.shape[0]))
        plt.figure(figsize=(16, 6))
        plt.plot(x, true_values[: len(x), idx], "k", label="Actual")
        plt.plot(x, mean_predictions[: len(x), idx], color="dodgerblue", linestyle="--", label="Predicted")
        plt.fill_between(
            x,
            lower_ci[: len(x), idx],
            upper_ci[: len(x), idx],
            alpha=0.3,
            color="dodgerblue",
            label="95% CI",
        )
        plt.xlabel("Time (minutes)")
        plt.ylabel(label)
        plt.title(f"Predictions with 95% CI — {label}")
        plt.grid()
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(output_dir / f"predictions_{label.lower()}.png")
        plt.close()


def compute_metrics(true_values, mean_predictions):
    return {
        "solar_r2": r2_score(true_values[:, 0], mean_predictions[:, 0]),
        "solar_mae": mean_absolute_error(true_values[:, 0], mean_predictions[:, 0]),
        "solar_rmse": np.sqrt(mean_squared_error(true_values[:, 0], mean_predictions[:, 0])),
        "wind_r2": r2_score(true_values[:, 1], mean_predictions[:, 1]),
        "wind_mae": mean_absolute_error(true_values[:, 1], mean_predictions[:, 1]),
        "wind_rmse": np.sqrt(mean_squared_error(true_values[:, 1], mean_predictions[:, 1])),
    }


# -----------------------------------------------LOAD DATA---------------------------------------------------------#
df = pd.read_csv('../../dataset/PSML.csv', parse_dates=['time'])
print(df.shape)
df.set_index('time', inplace=True)
df1 = df.ffill().bfill()

# ---------------------------------SPLIT, SCALE, AND CONVERT TO NP ARRAYS---------------------------------------#
X, y = feature_label_split(df1, targets=['solar_power', 'wind_power'], drop_cols=['load_power'])
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.20, 0.20, 0.60)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_arr = scaler_X.fit_transform(X_train)
X_val_arr = scaler_X.transform(X_val)
X_test_arr = scaler_X.transform(X_test)

y_train_arr = scaler_y.fit_transform(y_train)
y_val_arr = scaler_y.transform(y_val)
y_test_arr = scaler_y.transform(y_test)

# ---------------------------------DEFINE LOOKBACK AND FORMAT DATA------------------------------------------#
seq_length = 12

train_features_seq, train_targets_seq = create_sequences(torch.Tensor(X_train_arr), torch.Tensor(y_train_arr), seq_length)
val_features_seq, val_targets_seq = create_sequences(torch.Tensor(X_val_arr), torch.Tensor(y_val_arr), seq_length)
test_features_seq, test_targets_seq = create_sequences(torch.Tensor(X_test_arr), torch.Tensor(y_test_arr), seq_length)

print(f"Train features shape: {train_features_seq.shape}")
print(f"Train targets shape: {train_targets_seq.shape}")
print(f"Test features shape: {test_features_seq.shape}")
print(f"Test targets shape: {test_targets_seq.shape}")

# --------------------------STORE AS TENSORDATASET AND CREATE DATALOADERS-----------------------------#
train = TensorDataset(train_features_seq, train_targets_seq)
val = TensorDataset(val_features_seq, val_targets_seq)
test = TensorDataset(test_features_seq, test_targets_seq)

batch_size = 512
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    break

# ### Train and evaluate vGRU per fixed KL weight
num_layers = 1
lr = 0.001
hidden_size = 35
num_epochs = 50
n_samples = 50

results_root = Path(__file__).resolve().parent

for kl_weight in KL_WEIGHTS:
    weight_label = f"KL_{kl_weight:.0e}"
    case_dir = results_root / weight_label
    case_dir.mkdir(parents=True, exist_ok=True)

    model = GRUReparameterizationModel(
        in_features=8,
        hidden_size=hidden_size,
        out_features=2,
        num_layers=num_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reconstruction_loss_fn = torch.nn.MSELoss()

    start_train = time.time()
    train_losses, val_losses = train_model_with_fixed_kl_weight(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        reconstruction_loss_fn=reconstruction_loss_fn,
        optimizer=optimizer,
        kl_weight=kl_weight,
        device=device,
    )
    train_time = time.time() - start_train
    print(f"[{weight_label}] Training time: {train_time:.4f} seconds")

    save_loss_plot(
        train_losses,
        val_losses,
        case_dir / "train_val_loss.png",
        f"Training/Validation Loss ({weight_label})",
    )

    start_infer = time.time()
    mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(
        model,
        test_loader,
        n_samples=n_samples,
        scaler_y=scaler_y,
        device=device,
    )
    infer_time = time.time() - start_infer
    print(f"[{weight_label}] Inference time: {infer_time:.4f} seconds")

    inside_95 = ((true_values >= lower_ci) & (true_values <= upper_ci)).mean(axis=0)
    print(f"[{weight_label}] Empirical 95% coverage per output: {inside_95}")

    save_prediction_plots(
        mean_predictions,
        true_values,
        lower_ci,
        upper_ci,
        case_dir,
        n_display=43800 // 2,
    )

    metrics = compute_metrics(true_values, mean_predictions)
    metrics.update({
        "kl_weight": kl_weight,
        "train_time_sec": train_time,
        "infer_time_sec": infer_time,
        "coverage_solar": float(inside_95[0]),
        "coverage_wind": float(inside_95[1]),
    })

    pd.DataFrame([metrics]).to_csv(case_dir / "metrics.csv", index=False)

    save_arrays_to_csv(
        true_values,
        mean_predictions,
        lower_ci,
        upper_ci,
        case_dir / "vGRUTest.csv",
    )

print("Finished KL weight sensitivity evaluation.")
