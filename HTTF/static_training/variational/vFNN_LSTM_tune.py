from pathlib import Path

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
import optuna
import tqdm as notebook_tqdm
from data_handler import *
from uncertainty import *
from linear_variational import *

DATA_DIR = Path(__file__).resolve().parent

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(device))

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
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  # Use the last 'lookback' time steps for prediction
        y.append(data[i])  # The next time step is the target
    return np.array(X), np.array(y)

timesteps = 10
Xtrain, Ytrain = create_sequences(train_data, lookback=timesteps)
Xtest, Ytest = create_sequences(test_data, lookback=timesteps)
Xvalid, Yvalid = create_sequences(valid_data, lookback=timesteps)

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

class vLSTM(nn.Module):
    def __init__(self, in_features, hidden_size1, hidden_size2, hidden_size3, out_features, prior_mean=0, prior_variance=0.5, posterior_rho_init=-4.0, bias=True):
        super(vLSTM, self).__init__()

        # Define multiple LSTM layers
        self.lstm1 = nn.LSTM(in_features, hidden_size1, batch_first=True)

        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)

        self.lstm3 = nn.LSTM(hidden_size2, hidden_size3, batch_first=True)
        
        self.fc = LinearReparameterization(
            in_features=hidden_size3,
            out_features=out_features,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            posterior_rho_init=posterior_rho_init,
            bias=bias
        )

    def forward(self, x, hidden_states=None):
        # LSTM1
        out, _ = self.lstm1(x, hidden_states)
    
        # LSTM2
        out, _ = self.lstm2(out, hidden_states)

        # LSTM3
        out, _ = self.lstm3(out, hidden_states)

        # Get the output for the **last time step** only: [batch_size, hidden_size]
        hidden_last_step = out[:, -1, :]  # Last time step

        # Pass through the final linear layer
        output, kl_fc = self.fc(hidden_last_step)
        kl_total = kl_fc

        # Return output and total KL divergence
        return output, kl_total

# Define the Optuna objective function using your training loop
def objective(trial):
    # --- Hyperparameter suggestions ---
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    hidden_size1 = trial.suggest_int("hidden_size1", 16, 128, step=16)
    hidden_size2 = trial.suggest_int("hidden_size2", 16, 128, step=16)
    hidden_size3 = trial.suggest_int("hidden_size3", 8, 64, step=8)
    
    # Optionally, choose a KL schedule
    kl_schedule = "sigmoid_decay"

    # --- Instantiate the variational model ---
    # (Adjust in_features and out_features as needed for your data)
    in_features = 2
    out_features = 2
    model = vLSTM(
        in_features=in_features,
        hidden_size1=hidden_size1,
        hidden_size2=hidden_size2,
        hidden_size3=hidden_size3,
        out_features=out_features,
        prior_mean=0,
        prior_variance=0.5,
        posterior_rho_init=-4.0,
        bias=True
    ).to(device)

    # --- Optimizer and loss ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    reconstruction_loss_fn = nn.L1Loss()  # or nn.MSELoss()

    # --- DataLoaders for training and validation ---
    train_dataset = torch.utils.data.TensorDataset(Xtrain_tensor, Ytrain_tensor)
    val_dataset   = torch.utils.data.TensorDataset(Xvalid_tensor, Yvalid_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    num_epochs = 30
    # Track the validation loss from each epoch so that we can report to Optuna
    for epoch in range(num_epochs):
        model.train()

        # --- Update KL weight based on schedule ---
        if kl_schedule == 'linear':
            kl_weight = epoch / num_epochs
        elif kl_schedule == 'sigmoid_growth':
            kl_weight = 0.05 / (1 + np.exp(-2 * (epoch - 0.7 * num_epochs))) + 0.0005
        elif kl_schedule == 'sigmoid_decay':
            kl_weight = 0.05 / (1 + np.exp(2 * (epoch - 0.15 * num_epochs))) + 0.0005
        else:  # 'constant'
            kl_weight = 1.0

        running_train_loss = 0.0
        running_train_reconstruction_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()
            running_train_reconstruction_loss += reconstruction_loss.item()
            running_train_loss += total_loss.item()
        avg_train_loss = running_train_loss / len(train_loader)
        avg_train_reconstruction_loss = running_train_reconstruction_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        running_val_loss = 0.0
        running_val_reconstruction_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs, val_kl_loss = model(val_inputs)
                val_reconstruction_loss = reconstruction_loss_fn(val_outputs, val_targets)
                val_total_loss = val_reconstruction_loss + kl_weight * val_kl_loss
                running_val_reconstruction_loss += val_reconstruction_loss.item()
                running_val_loss += val_total_loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        avg_val_reconstruction_loss = running_val_reconstruction_loss / len(val_loader)

        print(f"Trial {trial.number}, Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, KL Weight: {kl_weight:.4f}, Train Recon. Loss: {avg_train_reconstruction_loss:.4f}, Val Recon. Loss: {avg_val_reconstruction_loss} ")

        # Report intermediate validation loss for pruning
        trial.report(avg_val_reconstruction_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the final validation loss (the lower the better)
    return avg_val_reconstruction_loss

# --- Run the study ---
trials = 20
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=trials)  # Use n_jobs>1 with caution on GPU

# Print out the best results
print("Best trial:")
best_trial = study.best_trial
print("  Final Validation Loss (MAE): {:.4f}".format(best_trial.value))
print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))
print()

trials_df = study.trials_dataframe()
print(trials_df)

trials_df

trials_df_sorted = trials_df.sort_values(by='value')
trials_df_sorted
