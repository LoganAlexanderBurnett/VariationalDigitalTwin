import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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

from httf.data_handler import create_autoregressive_sequences
from httf.models import DeterministicLSTMModel
from httf.predict import predict_deterministic
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


# Instantiate the model
batch_size = 512
model = DeterministicLSTMModel(input_size=2, hidden_size1=96, hidden_size2=16, hidden_size3=48, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000023)

# Training loop
epochs = 50
train_dataset = torch.utils.data.TensorDataset(Xtrain_tensor, Ytrain_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_dataset = torch.utils.data.TensorDataset(Xvalid_tensor, Yvalid_tensor)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

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

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(history['train_losses'], label='Training Loss')
plt.plot(history['val_losses'], label='Validation Loss')
plt.title(f"Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.savefig("training.png", bbox_inches='tight')
plt.close()

test_dataset = torch.utils.data.TensorDataset(Xtest_tensor, Ytest_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

Ypred_rescaled, Ytrue_rescaled = predict_deterministic(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
)

# Save the rescaled arrays as .npy files
np.save('Ytrue_rescaled.npy', Ytrue_rescaled)
np.save('Ypred_rescaled.npy', Ypred_rescaled)

# Extract individual outputs for evaluation
Ypred_1, Ypred_2 = Ypred_rescaled[:, 0], Ypred_rescaled[:, 1]
Ytrue_1, Ytrue_2 = Ytrue_rescaled[:, 0], Ytrue_rescaled[:, 1]

# Plot first output
plt.figure(figsize=(16, 6))

# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    plt.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
plt.plot(Ytrue_1, 'b', label="Actual")
plt.plot(Ypred_1, label="LSTM", color='r', linestyle='dashed')
plt.xlabel('Time Steps (30 seconds/step)')
plt.ylabel('Solid Temperature (°C)')
plt.ylim(0, 1200)
plt.xlim(0, test_length)
plt.legend(loc='upper left')
plt.savefig("testing_ts.png", bbox_inches='tight')
plt.close()

# Plot second output
plt.figure(figsize=(16, 6))

# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    plt.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.plot(Ytrue_2, 'b',label="Actual")
plt.plot(Ypred_2, label="LSTM", color='r', linestyle='dashed')
plt.xlabel('Time Steps (30 seconds/step)')
plt.ylabel('Fluid Temperature (°C)')
plt.ylim(0, 1200)
plt.xlim(0, test_length)
plt.legend(loc='upper left')
plt.savefig("testing_tf.png", bbox_inches='tight')
plt.close()

def calculate_metrics(y_true, y_pred):
    r2    = r2_score(y_true, y_pred)
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true)**2)) * 100
    return r2, mae, mape, rmse, rmspe

metrics = {}
for idx, (y_true, y_pred) in enumerate([(Ytrue_1, Ypred_1),
                                        (Ytrue_2, Ypred_2)], start=1):
    r2, mae, mape, rmse, rmspe = calculate_metrics(y_true, y_pred)

    print(f"--- Metrics for output #{idx} ---")
    print(f"R^2 Score: {r2}")
    print(f"MAE:        {mae}")
    print(f"MAPE:       {mape}%")
    print(f"RMSE:       {rmse}")
    print(f"RMSPE:      {rmspe}%\n")

    # cast to native floats for JSON
    metrics[f"output_{idx}"] = {
        "r2":    float(r2),
        "mae":   float(mae),
        "mape":  float(mape),
        "rmse":  float(rmse),
        "rmspe": float(rmspe)
    }

with open("performance_metrics.json", "w") as fp:
    json.dump(metrics, fp, indent=4)

print("Saved all metrics to performance_metrics.json")



