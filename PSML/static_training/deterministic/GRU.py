# Auto-generated from GRU.ipynb; edit the notebook if you need to regenerate this script.
# %% [markdown]
# # GRU forecasting of renewable energy grid production
# ## - Take mean of every 10 timesteps
# ## - 32,500 timesteps = 325,000 minutes = ~225.7 days: 60% training, 20% validation, 20% test
# ## - Lookback = 12 timesteps = 2hr
# ## - lr = 0.001, hidden_size = 35, num_epochs = 50, batch_size = 512
#
# ## Model Architecture
# ### 1 Linear input layer
# ### 1 GRU
# ### 2 Linear layers, 2nd to output

# %%
from psml.data_handler import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader

# %% [markdown]
# ### Set seed

# %%
import random

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

# %% [markdown]
# ### Define Standard GRU model

# %%
class StandardGRUModel(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, num_layers, bias=True):
        super(StandardGRUModel, self).__init__()

        # First Linear Layer
        self.fc1 = nn.Linear(in_features, hidden_size, bias=bias)
        
        # GRU Layers (num_layers stacked GRUs)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, 
                          num_layers=num_layers, batch_first=True, bias=bias)

        # Second Linear Layer
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=bias)

        # Final Output Layer
        self.fc3 = nn.Linear(hidden_size, out_features, bias=bias)

    def forward(self, x, hidden_states=None):
        # Pass input through the first linear layer
        x = self.fc1(x)
        x = F.relu(x)  # Apply ReLU activation
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size, device=x.device)
        
        # Pass through GRU layers, returning the output sequence and final hidden state
        hidden_seq, hidden_last = self.gru(x, hidden_states)

        # Take the last layer’s hidden state
        hidden_last = hidden_seq[:, -1, :]
        
        # Pass through second linear layer
        hidden_last = self.fc2(hidden_last)
        hidden_last = F.relu(hidden_last)  # Apply ReLU activation

        # Pass through the final output layer
        output = self.fc3(hidden_last)
        
        return output

# %% [markdown]
# ### Typical loading, scaling, and preparation of data

# %%
#-----------------------------------------------LOAD DATA---------------------------------------------------------#
df = pd.read_csv('../../dataset/PSML.csv', parse_dates=['time'])
df.set_index('time', inplace=True)
df1 = df.fillna(method='ffill').fillna(method='bfill')
print(df1.shape)

columns = df1.columns
print(columns)
print(df1.shape)

data = df1
data

# %%
#---------------------------------SPLIT, SCALE, AND CONVERT TO NP ARRAYS---------------------------------------#

# Use the function with a list of target columns
X, y = feature_label_split(data, targets=['solar_power', 'wind_power'], drop_cols=['load_power'])

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, train_fraction=0.80, validation_fraction=0.10, test_fraction=0.10)

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

# %% [markdown]
# ### Instantiate and train GRU model

# %%
# Training process (standard GRU model)
def train_gru(model, train_loader, val_loader, num_epochs, loss_fn, optimizer, device=torch.device('cpu')):
    model.to(device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero out the gradients from the optimizer
            optimizer.zero_grad()

            # Forward pass through the model
            outputs = model(inputs)

            # Compute the loss (no KL divergence in standard GRU model)
            loss = loss_fn(outputs, targets)

            # Backward pass to compute gradients
            loss.backward()

            # Optimizer step to update weights
            optimizer.step()

            running_train_loss += loss.item()

        # Compute average training loss
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation for validation
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                # Forward pass on validation data
                val_outputs = model(val_inputs)

                # Compute the validation loss
                val_loss = loss_fn(val_outputs, val_targets)
                running_val_loss += val_loss.item()

        # Compute average validation loss
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Print progress for each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses

# %%
# Hyperparameters from GridSearch
num_layers = 1
lr = 0.001
hidden_size = 35
num_epochs = 50

# Initialize model with current hyperparameters
model = StandardGRUModel(
    in_features=8,  # Adjust based on input features
    hidden_size=hidden_size,
    out_features=2,  # Adjust based on output features
    num_layers=num_layers
)

# Initialize optimizer with current learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Define loss function
reconstruction_loss_fn = torch.nn.MSELoss()  # regression task

# Timing
start = time.time()

# Train the model
train_losses, val_losses = train_gru(
    model, train_loader, val_loader, num_epochs=num_epochs, loss_fn=reconstruction_loss_fn, optimizer=optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

end = time.time()
train_time = end - start
print(f"Training time: {train_time:.4f} seconds")

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create the main plot
plt.figure(figsize=(5, 4))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend(loc='upper right')

last_epochs = 10

# Add a small inset zoomed view of the last 10 epochs
ax = plt.gca()  # Get the current axes
inset_ax = inset_axes(ax, width="40%", height="40%", loc="center right")  # Adjust as needed

# Plot the zoomed-in section on the inset
inset_ax.plot(range(len(train_losses) - last_epochs, len(train_losses)), train_losses[-last_epochs:], label="Training Loss")
inset_ax.plot(range(len(val_losses) - last_epochs, len(val_losses)), val_losses[-last_epochs:], label="Validation Loss")

# Save the figure (use any desired file path and format)
# plt.savefig("GRU_train_val_loss.png", dpi=300, bbox_inches='tight')

plt.show()

# %% [markdown]
# # Predict on test data

# %%
def predict_standard_gru(model, test_loader, scaler_y=None, device=torch.device('cpu')):
    model.eval()  # Set the model to evaluation mode
    
    all_predictions = []
    true_values = []

    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Store true values for comparison
            true_values.append(targets.cpu().numpy())

            # Generate predictions for each input
            outputs = model(inputs)

            # Store predictions
            all_predictions.append(outputs.cpu().numpy())

    # Concatenate predictions and true values across all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (total_samples, num_outputs)
    true_values = np.concatenate(true_values, axis=0)  # Shape: (total_samples, num_outputs)

    # Apply inverse scaling if scaler_y is provided
    if scaler_y is not None:
        true_values = scaler_y.inverse_transform(true_values)
        all_predictions = scaler_y.inverse_transform(all_predictions)

    return all_predictions, true_values

# %%
# Assuming you have a test DataLoader
start = time.time()
test_predictions, test_actuals = predict_standard_gru(model, test_loader, scaler_y=scaler_y, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
end = time.time()
print(f"Inference took {end-start}s.")

print("Test Predictions:", test_predictions.shape)
print("Test Actuals:", test_actuals.shape)

# %%
labels = ['Solar Power', 'Wind Power']

# Plot predictions vs actuals
for output_index, label in zip(range(0,2), labels):
    plt.figure(figsize=(16,6))
    plt.xlim(0, 13000*6)
    plt.plot(test_actuals[:, output_index], label='Actual', linestyle='-', color='blue')
    plt.plot(test_predictions[:, output_index], label='Prediction', color='r', linestyle='--')
    plt.xlabel('Time (10 minute intervals)')
    plt.ylabel(label)
    plt.title('GRU Test Predictions vs. Actuals')
    plt.legend()
    plt.grid(True)

plt.show()

# %% [markdown]
# # Calculate R2, MAE, RMSE

# %%
# Number of outputs
n_outputs = test_actuals.shape[1]

# Calculate MAPE and RMSPE for each output
r2_scores = [r2_score(test_actuals[:, i], test_predictions[:, i]) for i in range(n_outputs)]

mae_scores = [mean_absolute_error(test_actuals[:, i], test_predictions[:, i]) for i in range(n_outputs)]
rmse_scores = [np.sqrt(mean_squared_error(test_actuals[:, i], test_predictions[:, i])) for i in range(n_outputs)]

print(f"R2: {r2_scores}")
print(f"MAE: {mae_scores}")
print(f"RMSE: {rmse_scores}")

# %%
def save_arrays_to_csv(array1: np.ndarray, array2: np.ndarray, filename: str):
    # Create a DataFrame from the three arrays
    df = pd.DataFrame({
        'True Solar': array1[:, 0],
        'True Wind': array1[:, 1],
        'Predicted Solar': array2[:, 0],
        'Predicted Wind': array2[:, 1]
    })

    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)


save_arrays_to_csv(test_actuals, test_predictions, 'StandardGRUTest.csv')

# %% [markdown]
# # Predict on train

# %%
# Assuming you have a test DataLoader
train_predictions, train_actuals = predict_standard_gru(model, train_loader, scaler_y=scaler_y, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

print("Test Predictions:", train_predictions)
print("Test Actuals:", train_actuals)

# %%
labels = ['Solar Power', 'Wind Power']

# Plot predictions vs actuals
for output_index, label in zip(range(0,2), labels):
    plt.figure(figsize=(16,6))
    plt.xlim(0, 13000*6)
    plt.plot(train_actuals[:, output_index], label='Actual', linestyle='-', color='blue')
    plt.plot(train_predictions[:, output_index], label='Prediction', color='r', linestyle='--')
    plt.xlabel('Time (10 minute intervals)')
    plt.ylabel(label)
    plt.title('GRU Train Predictions vs. Actuals')
    plt.legend()
    plt.grid(True)

plt.show()

# %%
# Number of outputs
n_outputs = train_actuals.shape[1]

# Calculate MAPE and RMSPE for each output
r2_scores = [r2_score(train_actuals[:, i], train_predictions[:, i]) for i in range(n_outputs)]
mae_scores = [mean_absolute_error(train_actuals[:, i], train_predictions[:, i]) for i in range(n_outputs)]
rmse_scores = [np.sqrt(mean_squared_error(train_actuals[:, i], train_predictions[:, i])) for i in range(n_outputs)]

print(f"R2: {r2_scores}")
print(f"MAE: {mae_scores}")
print(f"RMSE: {rmse_scores}")

# %%

