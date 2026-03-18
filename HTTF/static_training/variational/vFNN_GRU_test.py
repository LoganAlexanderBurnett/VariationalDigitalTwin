from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time
import random
from torchinfo import summary

from HTTF.data_handler import create_autoregressive_sequences
from HTTF.trainer import train_model
from HTTF.uncertainty import predict_with_uncertainty
from HTTF.linear_variational import LinearReparameterization

DATA_DIR = Path(__file__).resolve().parent

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Load data
train = pd.read_csv(DATA_DIR / 'TF_TS_train.csv')
test = pd.read_csv(DATA_DIR / 'TF_TS_test.csv')
valid = pd.read_csv(DATA_DIR / 'TF_TS_valid.csv')

# Interpolate the NaN values
train['TS'] = train['TS'].interpolate()
test['TS'] = test['TS'].interpolate()
valid['TS'] = valid['TS'].interpolate()

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
Xtest,   Ytest: {Xtrain_tensor.shape}, {Ytrain_tensor.shape}
Xvalid, Yvalid: {Xvalid_tensor.shape}, {Yvalid_tensor.shape}"""
)

class vGRU(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, hidden_size3, out_features, prior_mean=0, prior_variance=1.0, posterior_rho_init=-3.0, bias=True
    ):
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

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary(model, input_size=(batch_size, timesteps, 2), device=device)

# Load data into datasets followed by dataloaders
train_dataset = torch.utils.data.TensorDataset(Xtrain_tensor, Ytrain_tensor)
valid_dataset = torch.utils.data.TensorDataset(Xvalid_tensor, Yvalid_tensor)
test_dataset = torch.utils.data.TensorDataset(Xtest_tensor, Ytest_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Investigating the shape of data in train_loader
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    
    # Break after inspecting the first batch
    break

# Timing
start = time.time()

# Train and validate the model
train_losses, val_losses = train_model(model, train_loader, valid_loader, num_epochs=num_epochs, reconstruction_loss_fn=criterion, optimizer=optimizer, kl_schedule=None, device=device)

end = time.time()
train_time = end - start
print(f"Training time on {torch.cuda.get_device_name()}: {train_time:.4f} seconds")

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create the main plot
plt.figure(figsize=(5, 4))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.yscale('log')
plt.legend(loc='upper right')

last_epochs = 10

# Add a small inset zoomed view of the last 10 epochs
ax = plt.gca()  # Get the current axes
inset_ax = inset_axes(ax, width="40%", height="40%", loc="center right")  # Adjust as needed

# Plot the zoomed-in section on the inset
inset_ax.plot(range(len(train_losses) - last_epochs, len(train_losses)), train_losses[-last_epochs:], label="Training Loss")
inset_ax.plot(range(len(val_losses) - last_epochs, len(val_losses)), val_losses[-last_epochs:], label="Validation Loss")

# Save the figure 
# plt.savefig("vGRU_train_val_loss.png", dpi=300, bbox_inches='tight')

plt.show()

# Function to make predictions multiple times to capture uncertainty

n_samples = 250

# Timing
start = time.time()

# Predict on test data, sampling to obtain uncertainty estimation, plot for each output
mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(model, test_loader, n_samples=n_samples, scaler_y=scaler, device=device)

n_predictions = true_values.shape[0] * true_values.shape[1] * n_samples

end = time.time()
cpu_time = end - start
print(f"Prediction time: {cpu_time:.4f} seconds for total of {n_predictions} predictions")

pred_determ = np.load('Ypred_rescaled.npy')

# Extract individual outputs for evaluation
pred_determ_ts, pred_determ_tf = pred_determ[:, 0], pred_determ[:, 1]

# Plot predictions vs actuals
plt.figure(figsize=(16, 6))

# Plot true values (ground truth)
plt.plot(true_values[:, 0], label='Actual', color='k', linewidth=2)

plt.plot(pred_determ_ts, label="GRU", color='b', linestyle='dashed')

# Plot predicted mean
plt.plot(mean_predictions[:, 0], label='Variational GRU Mean', color='darkcyan', linestyle='--')

N = len(mean_predictions)
x = np.arange(N)
plt.fill_between(
                x,
                lower_ci[:N, 0],
                upper_ci[:N, 0],
                alpha=0.3,
                color='darkcyan',
                label='95% CI'
            )

# Add vertical lines every 5485 time steps
for x in np.arange(5485, 340070, 5485):
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.ylabel('Temperature (°C)')
plt.xlabel('Time Steps (30 seconds/step)')
plt.xlim(0, 340070)
plt.legend()
#plt.grid(True)
plt.show()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sensor_range_i = 191975
sensor_range_f = sensor_range_i + 5473

# Create the figure and main axes explicitly
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(mean_predictions.shape[0])

# Plot true values (ground truth)
ax.plot(x, true_values[:, 0], label='Actual', color='k', linewidth=2)

ax.plot(pred_determ_ts, label="GRU", color='b', linestyle='--')

# Plot predicted mean
ax.plot(x, mean_predictions[:, 0], label='VGRU Mean', color='darkcyan', linestyle='--')

N = len(mean_predictions)
x = np.arange(N)
plt.fill_between(
                x,
                lower_ci[:N, 0],
                upper_ci[:N, 0],
                alpha=0.3,
                color='darkcyan',
                label='95% CI'
            )

# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax.set_ylabel('Solid Temperature (°C)')
ax.set_xlabel('Time Steps (30 seconds/step)')
ax.set_xlim(0, 340070)
leg = ax.legend(loc="upper left")  # save the legend as an artist

# Create an inset axis using the main axis 'ax' and specify its position with bbox_to_anchor.
axins = inset_axes(ax,
                   width=1.75, height=1.75,
                   bbox_to_anchor=(0.52, 0.65, 0.1, 0.1),  # adjust as needed
                   bbox_transform=ax.transAxes,
                   loc='center')

# Create a mask for the desired x-range (sensor_range_i to sensor_range_f)
mask = (x >= sensor_range_i) & (x <= sensor_range_f)

# Re-plot the same data on the inset
axins.plot(x[mask], true_values[mask, 0], color='k', linewidth=2)
axins.plot(x[mask], pred_determ_ts[mask], color='b', linestyle='--')
axins.plot(x[mask], mean_predictions[mask, 0], color='darkcyan', linestyle='--', linewidth=2)
axins.fill_between(
    x[mask],
    lower_ci[mask, 0],
    upper_ci[mask, 0],
    color='darkcyan', alpha=0.3
)

# Set the x-limits of the inset to zoom into the desired range
axins.set_xlim(sensor_range_i, sensor_range_f)

# Save the figure.
# Include extra artists (like the inset and legend) in bbox_extra_artists so that they are considered in the layout.
fig.savefig('VBBL_TS.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[axins, leg])
plt.show()

# Plot predictions vs actuals
plt.figure(figsize=(16, 6))
    
# Plot true values (ground truth)
plt.plot(true_values[:, 1], label='Actual', color='k', linewidth=2)

plt.plot(pred_determ_tf, label="GRU", color='b', linestyle='--')

# Plot predicted mean
plt.plot(mean_predictions[:, 1], label='Variational GRU Mean', color='darkcyan', linestyle='--')

# Plot confidence intervals (mean +/- confidence interval)
N = len(mean_predictions)
x = np.arange(N)
plt.fill_between(
                x,
                lower_ci[:N, 1],
                upper_ci[:N, 1],
                alpha=0.3,
                color='darkcyan',
                label='95% CI'
            )

# Add vertical lines every 5485 time steps
for x in np.arange(5485, 340070, 5485):
    plt.axvline(x=x, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

plt.ylabel('Temperature (°C)')
plt.xlabel('Time Steps (30 seconds/step)')
plt.xlim(0, 340070)
plt.legend()
#plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sensor_range_i = 87760
sensor_range_f = sensor_range_i + 5473

# Create the figure and main axes explicitly
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(mean_predictions.shape[0])

# Plot true values (ground truth)
ax.plot(x, true_values[:, 1], label='Actual', color='k', linewidth=2)

ax.plot(pred_determ_tf, label="GRU", color='b', linestyle='--')

# Plot predicted mean
ax.plot(x, mean_predictions[:, 1], label='VGRU Mean', color='darkcyan', linestyle='--')

# Plot confidence intervals (mean +/- confidence interval)
N = len(mean_predictions)
x = np.arange(N)
ax.fill_between(
                x,
                lower_ci[:N, 1],
                upper_ci[:N, 1],
                alpha=0.3,
                color='darkcyan',
                label='95% CI'
            )
# Add vertical lines every 5485 time steps in the main plot
for x_val in np.arange(5485, 340070, 5485):
    ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

ax.set_ylabel('Fluid Temperature (°C)')
ax.set_xlabel('Time Steps (30 seconds/step)')
ax.set_xlim(0, 340070)
leg = ax.legend(loc="upper left")  # save the legend as an artist

# Create an inset axis using the main axis 'ax' and specify its position with bbox_to_anchor.
axins = inset_axes(ax,
                   width=1.75, height=1.75,
                   bbox_to_anchor=(0.34, 0.75, 0.1, 0.1),  # adjust as needed
                   bbox_transform=ax.transAxes,
                   loc='center')

# Create a mask for the desired x-range (sensor_range_i to sensor_range_f)
mask = (x >= sensor_range_i) & (x <= sensor_range_f)

# Re-plot the same data on the inset
axins.plot(x[mask], true_values[mask, 1], color='k', linewidth=2)
axins.plot(x[mask], pred_determ_tf[mask], color='b', linestyle='--')
axins.plot(x[mask], mean_predictions[mask, 1], color='darkcyan', linestyle='--')
axins.fill_between(
    x[mask],
    lower_ci[mask, 1],
    upper_ci[mask, 1],
    color='darkcyan', alpha=0.3
)

# Set the x-limits of the inset to zoom into the desired range
axins.set_xlim(sensor_range_i, sensor_range_f)

# Save the figure.
# Include extra artists (like the inset and legend) in bbox_extra_artists so that they are considered in the layout.
fig.savefig('VBBL_TF.png', dpi=300, bbox_inches='tight', bbox_extra_artists=[axins, leg])
plt.show()

calculate_and_display_metrics(true_values, mean_predictions)

import numpy as np

def compute_mape_per_output(true_values: np.ndarray, pred_values: np.ndarray) -> np.ndarray:
    """
    Compute MAPE for each output column:
      MAPE_j = 100 * mean(|true[:, j] - pred[:, j]| / |true[:, j]|)
    Ignores any zero entries in true to avoid division by zero.

    Args:
      true_values: shape (N, 2)
      pred_values: shape (N, 2)

    Returns:
      mape_per_output: shape (2,), with MAPE for each column
    """
    m, k = true_values.shape
    mape_per_output = np.zeros(k, dtype=float)

    for j in range(k):
        t_col = true_values[:, j]
        p_col = pred_values[:, j]
        nonzero_mask = t_col != 0
        if np.any(nonzero_mask):
            abs_perc_err = np.abs(t_col[nonzero_mask] - p_col[nonzero_mask]) / np.abs(t_col[nonzero_mask])
            mape_per_output[j] = 100.0 * np.mean(abs_perc_err)
        else:
            mape_per_output[j] = np.nan  # or some sentinel

    return mape_per_output



mape_TS, mape_TF = compute_mape_per_output(true_values, mean_predictions)
print(f"MAPE (TS) = {mape_TS:.4f}%")
print(f"MAPE (TF) = {mape_TF:.4f}%")
