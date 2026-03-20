# # vGRU forecasting of renewable energy grid production
# ## - Lookback = 12 timesteps = 2hr
# ## - lr = 0.001, hidden_size = 35, num_epochs = 50, batch_size = 512
#
# ## Model Architecture
# ### 1 Linear input layer
# ### 1 GRU
# ### 1 Linear layer
# ### 1 vLinear layer to output

# ### Import packages, set seed, define model

from psml.data_handler import *
from psml.trainer import *
from psml.predict import calculate_and_display_metrics, plot_predictions, predict_with_uncertainty
from psml.linear_variational import *
from psml.models import GRUReparameterizationModel
import torch
import torch.nn as nn
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader

set_random_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


#-----------------------------------------------LOAD DATA---------------------------------------------------------#
df = pd.read_csv('../../dataset/PSML.csv', parse_dates=['time'])
print(df.shape)
df.set_index('time', inplace=True)
df1 = df.ffill().bfill()

#---------------------------------SPLIT, SCALE, AND CONVERT TO NP ARRAYS---------------------------------------#

# Use the function with a list of target columns
X, y = feature_label_split(df1, targets=['solar_power', 'wind_power'], drop_cols=['load_power'])  # Predicting renewable energy production, not overall grid demand

# Split data
#X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_fraction=0.20, validation_fraction=0.20, test_fraction=0.60)
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, 0.2, 0.2, 0.6)

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_arr = scaler_X.fit_transform(X_train)
X_val_arr = scaler_X.transform(X_val)       # All as numpy.nd arrays
X_test_arr = scaler_X.transform(X_test)

y_train_arr = scaler_y.fit_transform(y_train)
y_val_arr = scaler_y.transform(y_val)
y_test_arr = scaler_y.transform(y_test)

#---------------------------------DEFINE LOOKBACK AND FORMAT DATA------------------------------------------#

# Define the sequence length
seq_length = 12

# Convert training, validation, and test sets into sequences
train_features_seq, train_targets_seq = create_sequences(torch.Tensor(X_train_arr), torch.Tensor(y_train_arr), seq_length)
val_features_seq, val_targets_seq = create_sequences(torch.Tensor(X_val_arr), torch.Tensor(y_val_arr), seq_length)
test_features_seq, test_targets_seq = create_sequences(torch.Tensor(X_test_arr), torch.Tensor(y_test_arr), seq_length)

# Check the new shapes of the data
print(f"Train features shape: {train_features_seq.shape}")
print(f"Train targets shape: {train_targets_seq.shape}")
print(f"Test features shape: {test_features_seq.shape}")
print(f"Test targets shape: {test_targets_seq.shape}")

#--------------------------STORE AS TENSORDATASET AND CREATE DATALOADERS-----------------------------#

# Create TensorDatasets
train = TensorDataset(train_features_seq, train_targets_seq)
val = TensorDataset(val_features_seq, val_targets_seq)
test = TensorDataset(test_features_seq, test_targets_seq)

# Create DataLoaders
batch_size = 512
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

# Investigating the shape of data in train_loader
for batch_idx, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {batch_idx+1}:")
    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shape: {targets.shape}")
    
    # Break after inspecting the first batch
    break

# ### Instantiate and train vGRU model

# Hyperparameters from GridSearch
num_layers = 1
lr = 0.001
hidden_size = 35
num_epochs = 50

# Initialize model with current hyperparameters
model = GRUReparameterizationModel(
    in_features=8,  # Adjust based on input features
    hidden_size=hidden_size,
    out_features=2,  # Adjust based on output features
    num_layers=num_layers
)

# Initialize optimizer with learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define loss function
reconstruction_loss_fn = torch.nn.MSELoss()  # regression task

# Timing
start = time.time()

# Train and validate the model
train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=num_epochs, reconstruction_loss_fn=reconstruction_loss_fn, optimizer=optimizer, device=device)

end = time.time()
train_time = end - start
print(f"Training time: {train_time:.4f} seconds")

# Create the main plot
plt.figure(figsize=(5, 4))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend(loc='upper right')
plt.show()

# # Predict on test data

n_samples = 50

# Timing
start = time.time()

# Predict on test data, sampling to obtain uncertainty estimation, plot for each output
mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(model, test_loader, n_samples=n_samples, scaler_y=scaler_y, device=device)

end = time.time()
infer_time = end - start
print(f"Inference time: {infer_time:.4f} seconds")

# say your stored 95% interval is [lower_ci, upper_ci] from predict_with_uncertainty
inside_95 = ((true_values >= lower_ci) & (true_values <= upper_ci)).mean(axis=0)
print("Empirical 95% coverage per output:", inside_95)

plot_predictions(
    mean_predictions,
    true_values,
    n_display=43800//2,
    lower=lower_ci,
    upper=upper_ci
)

# # Calculate R2, MAE, RMSE

calculate_and_display_metrics(true_values, mean_predictions)

def save_arrays_to_csv(
    true_values: np.ndarray,
    mean_predictions: np.ndarray,
    lower_ci: np.ndarray,
    upper_ci: np.ndarray,
    filename: str
):
    """
    Saves true values, predicted means, and prediction intervals to a CSV.
    
    Args:
        true_values      (np.ndarray): shape (N, 2) for solar and wind true values.
        mean_predictions (np.ndarray): shape (N, 2) for solar and wind predicted means.
        lower_ci         (np.ndarray): shape (N, 2) for solar and wind lower confidence bounds.
        upper_ci         (np.ndarray): shape (N, 2) for solar and wind upper confidence bounds.
        filename         (str): path to the output CSV file.
    """
    df = pd.DataFrame({
        'True Solar':            true_values[:, 0],
        'True Wind':             true_values[:, 1],
        'Predicted Mean Solar':  mean_predictions[:, 0],
        'Predicted Mean Wind':   mean_predictions[:, 1],
        'Lower CI Solar':        lower_ci[:, 0],
        'Lower CI Wind':         lower_ci[:, 1],
        'Upper CI Solar':        upper_ci[:, 0],
        'Upper CI Wind':         upper_ci[:, 1],
    })
    df.to_csv(filename, index=False)


save_arrays_to_csv(true_values, mean_predictions, lower_ci, upper_ci, 'vGRUTest.csv')


