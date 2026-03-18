from pathlib import Path
import sys

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "HTTF").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from HTTF.data_handler import create_autoregressive_sequences

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

DATA_DIR = Path(__file__).resolve().parent

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        self.fc = nn.Linear(hidden_size3, output_size)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out

# Instantiate the model
batch_size = 512
model = LSTMModel(input_size=2, hidden_size1=96, hidden_size2=16, hidden_size3=48, output_size=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.000023)

summary(model, input_size=(batch_size, timesteps, 2), device=device)

# Training loop
epochs = 50
train_dataset = torch.utils.data.TensorDataset(Xtrain_tensor, Ytrain_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

valid_dataset = torch.utils.data.TensorDataset(Xvalid_tensor, Yvalid_tensor)
valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

start = time.time()

history = {'loss': [], 'val_loss': []}
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    #scheduler.step()
    history['loss'].append(epoch_loss / len(train_loader))
    loss = epoch_loss / len(train_loader)
    
    # Validation loss
    model.eval()
    val_loss_accum = 0.0
    with torch.no_grad():
        for Xv_batch, Yv_batch in valid_loader:
            Xv_batch = Xv_batch.to(device)
            Yv_batch = Yv_batch.to(device)

            val_out = model(Xv_batch)
            batch_val_loss = criterion(val_out, Yv_batch)
            val_loss_accum += batch_val_loss.item()

    avg_val_loss = val_loss_accum / len(valid_loader)
    history['val_loss'].append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss:.4f}, Val Loss: {avg_val_loss:.4f}")


end = time.time()
print(f"Training took {end - start}s")

# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title(f"Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.grid()
plt.show()

test_dataset = torch.utils.data.TensorDataset(Xtest_tensor, Ytest_tensor)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2) Run a small loop over test batches to collect predictions and compute test loss
model.eval()
test_loss_accum = 0.0
all_preds = []
all_trues = []

start = time.time()
with torch.no_grad():
    for X_batch, Y_batch in test_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)

        # Compute batch loss (e.g. MSE)
        loss = criterion(outputs, Y_batch)
        test_loss_accum += loss.item()

        # Store predictions and true values (move back to CPU)
        all_preds.append(outputs.cpu())
        all_trues.append(Y_batch.cpu())
        
# 4) Concatenate all batch‐wise tensors into single tensors
all_preds = torch.cat(all_preds, dim=0)  # shape: (num_test_samples, ...)
all_trues = torch.cat(all_trues, dim=0)

# Convert to NumPy for plotting
Ypred = all_preds.cpu().numpy()
Ytrue = all_trues.cpu().numpy()

# Rescale predictions and true values back to their original scale
Ypred_rescaled = scaler.inverse_transform(Ypred)
Ytrue_rescaled = scaler.inverse_transform(Ytrue)

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
