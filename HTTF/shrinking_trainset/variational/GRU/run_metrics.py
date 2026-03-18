import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
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

from HTTF.data_handler import create_autoregressive_sequences
from HTTF.linear_variational import LinearReparameterization
from HTTF.trainer import train_model
from HTTF.uncertainty import predict_with_uncertainty

def set_random_seed(seed_value=42):
    # Python random seed
    random.seed(seed_value)
    
    # Numpy random seed
    np.random.seed(seed_value)
    
    # PyTorch seed
    torch.manual_seed(seed_value)
    
    # If using CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
        torch.backends.cudnn.deterministic = True  # For reproducibility
        torch.backends.cudnn.benchmark = False  # Disable auto-optimization for determinism

set_random_seed()

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

# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
valid = pd.read_csv('valid.csv')

test_length = len(test)

# Function to find rows with NaN values in a pandas DataFrame
def check_nans_in_dataframe(df, column_name, name):
    nan_rows = df[df[column_name].isna()]
    if not nan_rows.empty:
        print(f"Rows with NaN values in {name} dataset:")
        print(nan_rows)
    else:
        print(f"No NaN values in {name} dataset.")

# Check for NaN values in each dataset
check_nans_in_dataframe(train, 'TS', 'Train')
check_nans_in_dataframe(test, 'TS', 'Test')
check_nans_in_dataframe(valid, 'TS', 'Validation')

# Preprocess the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_data = scaler.fit_transform(train)
test_data = scaler.transform(test)
valid_data = scaler.transform(valid)

# Create sequences
timesteps = 10
Xtrain, Ytrain = create_autoregressive_sequences(train_data, lookback=timesteps)
Xtest, Ytest = create_autoregressive_sequences(test_data, lookback=timesteps)
Xvalid, Yvalid = create_autoregressive_sequences(valid_data, lookback=timesteps)

# Convert to PyTorch tensors
Xtrain_tensor = torch.tensor(Xtrain, dtype=torch.float32).to(device)
Ytrain_tensor = torch.tensor(Ytrain, dtype=torch.float32).to(device)
Xtest_tensor = torch.tensor(Xtest, dtype=torch.float32).to(device)
Ytest_tensor = torch.tensor(Ytest, dtype=torch.float32).to(device)
Xvalid_tensor = torch.tensor(Xvalid, dtype=torch.float32).to(device)
Yvalid_tensor = torch.tensor(Yvalid, dtype=torch.float32).to(device)

print(
    f"""Xtrain, Ytrain: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xtest,   Ytest: {Xtest_tensor.shape}, {Ytest_tensor.shape}
Xvalid, Yvalid: {Xvalid_tensor.shape}, {Yvalid_tensor.shape}"""
)


# Define the GRU model
class vGRU(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, hidden_size3, out_features, prior_mean=0, prior_variance=1.0, posterior_rho_init=-3.0, bias=True):
        super(vGRU, self).__init__()

        # Define multiple GRU layers
        self.gru1 = nn.GRU(in_features, hidden_size1, batch_first=True)

        self.gru2 = nn.GRU(hidden_size1, hidden_size2, batch_first=True)

        self.gru3 = nn.GRU(hidden_size2, hidden_size3, batch_first=True)

        self.fc = LinearReparameterization(
            in_features=hidden_size3,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def forward(self, x):

        # — LAYER 1 — 
        out, _ = self.gru1(x)

        # — LAYER 2 — 
        out, _ = self.gru2(out)

        # — LAYER 3 — 
        out, _ = self.gru3(out)

        # We only need the last time step's features:
        hidden_last_step = out[:, -1, :]  # → (batch, hidden_size3)

        # — FINAL VARIATIONAL LAYER —
        output, kl_fc = self.fc(hidden_last_step)
        kl_total = kl_fc

        return output, kl_total


# Instantiate the model
batch_size = 256
learning_rate = 0.00018
num_epochs = 50
hidden1 = 48
hidden2 = 64
hidden3 = 32
criterion = nn.MSELoss()

model = vGRU(in_features=2, hidden_size1=hidden1, hidden_size2=hidden2, hidden_size3=hidden3, out_features=2).to(device)
for gru in (model.gru1, model.gru2, model.gru3):
    gru.flatten_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Load data into datasets followed by dataloaders
train_dataset = torch.utils.data.TensorDataset(Xtrain_tensor, Ytrain_tensor)
valid_dataset = torch.utils.data.TensorDataset(Xvalid_tensor, Yvalid_tensor)
test_dataset = torch.utils.data.TensorDataset(Xtest_tensor, Ytest_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)


start = time.time()

# Train and validate the model
train_losses, val_losses = train_model(model, train_loader, valid_loader, num_epochs=num_epochs, reconstruction_loss_fn=criterion, optimizer=optimizer, kl_schedule=None, device=device)

end = time.time()
train_time = end - start
print(f"Training took {train_time}s")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title(f"Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.savefig("training.png", bbox_inches='tight')
plt.close()


n_samples = 100

# Timing
start = time.time()

# Predict on test data, sampling to obtain uncertainty estimation, plot for each output
mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(model, test_loader, n_samples=n_samples, scaler_y=scaler, device=device)

end = time.time()
cpu_time = end - start
print(f"Prediction time: {cpu_time:.4f} seconds")

Ypred = np.stack((mean_predictions, lower_ci, upper_ci), axis=2)

# Save Ypred and Ytrue
np.save("Ypred.npy", Ypred)
np.save("Ytrue.npy", true_values)

print("Saved Ypred.npy with shape:", Ypred.shape)
print("Saved Ytrue.npy with shape:", true_values.shape)

# Extract individual outputs for evaluation
Ypred_mean_1, Ypred_mean_2 = mean_predictions[:, 0], mean_predictions[:, 1]
Ypred_upper_1, Ypred_upper_2 = upper_ci[:, 0], upper_ci[:, 1]
Ypred_lower_1, Ypred_lower_2 = lower_ci[:, 0], lower_ci[:, 1]
Ytrue_1, Ytrue_2 = true_values[:, 0], true_values[:, 1]


def plot_with_ci(y_true, y_pred_mean, y_pred_lower, y_pred_upper,
                 ylabel, fname, test_length):
    plt.figure(figsize=(16, 6))

    # vertical lines every 5485 steps
    for x_val in np.arange(5485, len(y_true), 5485):
        plt.vlines(x=x_val, ymin=-100, ymax=1500,
                   color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    # actual and mean prediction
    x = np.arange(len(y_true))
    plt.plot(x, y_true, 'b', label="Actual")
    plt.plot(x, y_pred_mean, 'r', linestyle='--', label="Mean Prediction")

    # confidence interval band
    plt.fill_between(x, y_pred_lower, y_pred_upper,
                     color='r', alpha=0.2, label="95% CI")

    plt.xlabel('Time Steps (30 seconds/step)')
    plt.ylabel(ylabel)
    plt.ylim(0, 1200)
    plt.xlim(0, test_length)
    plt.legend(loc='upper left')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()


plot_with_ci(
    Ytrue_1,
    Ypred_mean_1,
    Ypred_lower_1,
    Ypred_upper_1,
    ylabel='Solid Temperature (°C)',
    fname='testing_ts.png',
    test_length=test_length
)

plot_with_ci(
    Ytrue_2,
    Ypred_mean_2,
    Ypred_lower_2,
    Ypred_upper_2,
    ylabel='Fluid Temperature (°C)',
    fname='testing_tf.png',
    test_length=test_length
)

def calculate_metrics(y_true, y_pred):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true)**2)) * 100
    return r2, mae, mape, rmse, rmspe

metrics = {}
for idx, (y_true, y_pred) in enumerate([
        (Ytrue_1, Ypred_mean_1),
        (Ytrue_2, Ypred_mean_2)],
    start=1):
    r2, mae, mape, rmse, rmspe = calculate_metrics(y_true, y_pred)

    print(f"--- Metrics for output #{idx} ---")
    print(f"R^2 Score: {r2}")
    print(f"MAE:        {mae}")
    print(f"MAPE:       {mape:.2f}%")
    print(f"RMSE:       {rmse}")
    print(f"RMSPE:      {rmspe:.2f}%\n")

    metrics[f"output_{idx}"] = {
        "r2":    float(r2),
        "mae":   float(mae),
        "mape":  float(mape),
        "rmse":  float(rmse),
        "rmspe": float(rmspe),
        "train_time": float(train_time)
    }

with open("performance_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=4)

print("Saved all metrics to performance_metrics.json")



