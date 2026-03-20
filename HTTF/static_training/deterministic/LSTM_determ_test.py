from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf.data_handler import (
    build_tensor_dataloader,
    prepare_autoregressive_splits,
    report_nan_rows,
)
from httf.models import DeterministicLSTMModel
from httf.predict import predict_deterministic
from httf.trainer import set_random_seed, train_deterministic

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
from torchinfo import summary

DATA_DIR = Path(__file__).resolve().parent

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

set_random_seed()

# Load data
train = pd.read_csv(DATA_DIR / 'TF_TS_train.csv')
test = pd.read_csv(DATA_DIR / 'TF_TS_test.csv')
valid = pd.read_csv(DATA_DIR / 'TF_TS_valid.csv')

# Check for NaN values in each dataset
report_nan_rows(train, 'TS', 'Train')
report_nan_rows(test, 'TS', 'Test')
report_nan_rows(valid, 'TS', 'Validation')

# Preprocess the data
timesteps = 10
prepared = prepare_autoregressive_splits(
    train,
    test,
    valid,
    lookback=timesteps,
    device=device,
    interpolate_columns=['TS'],
)
scaler = prepared["scaler"]
Xtrain_tensor, Ytrain_tensor = prepared["train"]
Xtest_tensor, Ytest_tensor = prepared["test"]
Xvalid_tensor, Yvalid_tensor = prepared["valid"]

print(
    f"""Xtrain, Ytrain: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xtest,   Ytest: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xvalid, Yvalid: {Xvalid_tensor.shape}, {Yvalid_tensor.shape}"""
)

# Instantiate the model
batch_size = 512
model = DeterministicLSTMModel(input_size=2, hidden_size1=96, hidden_size2=16, hidden_size3=48, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000023)

summary(model, input_size=(batch_size, timesteps, 2), device=device)

# Training loop
epochs = 50
train_loader = build_tensor_dataloader(Xtrain_tensor, Ytrain_tensor, batch_size=batch_size, shuffle=False)
valid_loader = build_tensor_dataloader(Xvalid_tensor, Yvalid_tensor, batch_size=batch_size, shuffle=False)

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
plt.show()

test_loader = build_tensor_dataloader(Xtest_tensor, Ytest_tensor, batch_size=batch_size, shuffle=False)

start = time.time()
Ypred_rescaled, Ytrue_rescaled = predict_deterministic(
    model,
    test_loader,
    scaler_y=scaler,
    device=device,
)
end = time.time()
print(f"Inference took: {end-start:.4f}s")

# Save the rescaled arrays as .npy files
# np.save('Ytrue_rescaled.npy', Ytrue_rescaled)
# np.save('Ypred_rescaled.npy', Ypred_rescaled)

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
plt.xlim(0, 340070)
plt.legend(loc='upper left')
plt.show()

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
plt.xlim(0, 340070)
plt.legend(loc='upper left')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define the x-range for the inset
sensor_range_i = 257795
sensor_range_f = sensor_range_i + 5473

# Create the figure and main axes
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(Ytrue_1))  # assuming Ytrue_1 and Ypred_1 have the same length

# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    ax.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot the main data
ax.plot(Ytrue_1, 'b', label="Actual")
ax.plot(Ypred_1, label="LSTM", color='r', linestyle='dashed')
ax.set_xlabel('Time Steps (30 seconds/step)')
ax.set_ylabel('Solid Temperature (°C)')
ax.set_ylim(0, 1200)
ax.set_xlim(0, 340070)
leg = ax.legend(loc='upper left')

# Create an inset axis using the same bbox settings as before
axins = inset_axes(ax,
                   width=1.75, height=1.75, 
                   bbox_to_anchor=(0.52, 0.65, 0.1, 0.1),  # adjust as needed,
                   bbox_transform=ax.transAxes,
                   loc='center')

# Create a mask for the desired x-range for the inset plot
mask = (x >= sensor_range_i) & (x <= sensor_range_f)

# Re-plot the same data on the inset for the zoomed region
axins.plot(x[mask], Ytrue_1[mask], 'b')
axins.plot(x[mask], Ypred_1[mask], color='r', linestyle='dashed')
# Optionally, add vertical lines to the inset if they fall within the range
for x_val in np.arange(5485, 340070, 5485):
    if sensor_range_i <= x_val <= sensor_range_f:
        axins.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Set the x-limits of the inset to zoom in to the desired region
axins.set_xlim(sensor_range_i, sensor_range_f)
# Optionally, adjust y-limits on the inset if needed, e.g.:
# axins.set_ylim(lower_bound, upper_bound)
fig.savefig('DETERM_TS.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[axins, leg])
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Define the sensor range for the inset
sensor_range_i = 257795
sensor_range_f = sensor_range_i + 5473

# Create the figure and main axes for the second output
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(Ytrue_2))  # Assuming Ytrue_2 and Ypred_2 have the same length

# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    ax.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Plot the main data for the second output
ax.plot(Ytrue_2, 'b', label="Actual")
ax.plot(Ypred_2, label="LSTM", color='r', linestyle='dashed')
ax.set_xlabel('Time Steps (30 seconds/step)')
ax.set_ylabel('Fluid Temperature (°C)')
ax.set_ylim(0, 1200)
ax.set_xlim(0, 340070)
leg = ax.legend(loc='upper left')

# Create an inset axis using the same positioning parameters as before
axins = inset_axes(ax,
                   width=1.75, height=1.75, 
                   bbox_to_anchor=(0.47, 0.75, 0.1, 0.1),
                   bbox_transform=ax.transAxes,
                   loc='center')

# Create a mask for the desired x-range for the inset plot
mask = (x >= sensor_range_i) & (x <= sensor_range_f)

# Re-plot the same data on the inset for the zoomed region
axins.plot(x[mask], Ytrue_2[mask], 'b')
axins.plot(x[mask], Ypred_2[mask], color='r', linestyle='dashed')
# Optionally add vertical lines in the inset if they fall within the range
for x_val in np.arange(5485, 340070, 5485):
    if sensor_range_i <= x_val <= sensor_range_f:
        axins.vlines(x=x_val, ymin=-100, ymax=1500, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

# Set the x-limits of the inset to zoom into the desired region
axins.set_xlim(sensor_range_i, sensor_range_f)
fig.savefig('DETERM_TF.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[axins, leg])
plt.show()

# Calculate metrics
def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe = np.sqrt(np.mean(((y_true - y_pred) / y_true)**2)) * 100
    return r2, mae, mape, rmse, rmspe

r2, mae, mape, rmse, rmspe = calculate_metrics(Ytrue_1, Ypred_1)
print(f"R^2 Score: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"RMSE: {rmse}")
print(f"RMSPE: {rmspe}%")
print()
r2, mae, mape, rmse, rmspe = calculate_metrics(Ytrue_2, Ypred_2)
print(f"R^2 Score: {r2}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}%")
print(f"RMSE: {rmse}")
print(f"RMSPE: {rmspe}%")
