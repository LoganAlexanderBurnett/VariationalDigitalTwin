import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from psml.data_handler import feature_label_split, create_sequences
from psml.linear_variational import LinearReparameterization
from psml.models import LSTMReparameterizationModel

# -----------------------------------------------------------------------------
# 1) Reproducibility & device
# -----------------------------------------------------------------------------
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_random_seed()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# -----------------------------------------------------------------------------
# 2) Model definition
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 3) Helpers
# -----------------------------------------------------------------------------
def train_model(model, train_loader, num_epochs, reconstruction_loss_fn, optimizer, device=torch.device('cpu'), kl_schedule=None):
    model.to(device)
    
    train_losses = []

    for epoch in range(num_epochs):
        model.train()

        if kl_schedule == 'linear':
            kl_weight = epoch / num_epochs
        elif kl_schedule == 'sigmoid_growth':
            kl_weight = 0.1 / (1 + np.exp(-2 * (epoch - 0.7 * num_epochs))) + 0.001 # max / (1 + e^[-rate * (epoch - frac_training_w/o_KL*num_epochs)]) + min
        elif kl_schedule == 'sigmoid_decay':
            kl_weight = 0.1 / (1 + np.exp(2 * (epoch - 0.15 * num_epochs))) + 0.001
        else: 
            kl_weight = 1e-4
        
        running_train_loss = 0.0
        running_recon_loss = 0.0
        running_kl_loss    = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs, kl_loss = model(inputs)

            # Compute the reconstruction loss
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)

            # Total loss (reconstruction + KL divergence)
            total_loss = reconstruction_loss + kl_weight * kl_loss

            # Backward pass
            total_loss.backward()
            optimizer.step()

            # Accumulate
            running_train_loss += total_loss.item()
            running_recon_loss += reconstruction_loss.item()
            running_kl_loss    += kl_loss.item()

        # Compute per‐batch averages
        n_batches = len(train_loader)
        avg_recon = running_recon_loss / n_batches
        avg_kl    = running_kl_loss    / n_batches
        avg_train_loss = running_train_loss / n_batches
        train_losses.append(avg_train_loss)

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, MSE Loss: {avg_recon:.6f},KL Loss: {avg_kl:.6f}')

    return train_losses


from joblib import Parallel, delayed

def predict_with_uncertainty(
    model,
    test_loader,
    n_samples=100,
    scaler_y=None,
    device=torch.device('cpu'),
    n_jobs=4,
    alpha=0.05
):
    """
    Runs MC sampling through `model` to estimate uncertainty.
    
    Returns:
      mean_preds: (N, K) array of predictive means
      true_vals:  (N, K) array of ground truths
      lower:      (N, K) lower bound at alpha/2 (default 2.5%)
      upper:      (N, K) upper bound at 1-alpha/2 (default 97.5%)
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_trues.append(targets.cpu().numpy())
            
            # sample forward passes in parallel
            def sample_once():
                outs, _ = model(inputs)
                return outs.detach().cpu().numpy()
            
            batch_preds = Parallel(n_jobs=n_jobs)(
                delayed(sample_once)() for _ in range(n_samples)
            )  # list of (batch, K) arrays
            
            # stack into shape (n_samples, batch, K)
            all_preds.append(np.stack(batch_preds, axis=0))
    
    # stack across batches → (n_samples, total_N, K)
    all_preds = np.concatenate(all_preds, axis=1)
    true_vals = np.concatenate(all_trues, axis=0)  # (total_N, K)
    
    # inverse-transform if needed
    if scaler_y is not None:
        true_vals = scaler_y.inverse_transform(true_vals)
        all_preds = np.array([scaler_y.inverse_transform(p) for p in all_preds])
    
    # compute statistics over the sample axis
    mean_preds = np.mean(all_preds, axis=0)  # (total_N, K)
    lower = np.percentile(all_preds, 100 * (alpha/2), axis=0)
    upper = np.percentile(all_preds, 100 * (1 - alpha/2), axis=0)
    
    return mean_preds, true_vals, lower, upper


def plot_predictions(
    preds,
    trues,
    title,
    labels=['Solar','Wind'],
    n_display=500,
    lower=None,
    upper=None
):
    """
    preds:        (T, K) array of predictive means
    trues:        (T, K) array of ground‐truths
    lower, upper: optional (T, K) arrays of UQ bounds (e.g. 2.5% and 97.5% quantiles)
    """
    N = min(len(preds), n_display)
    x = np.arange(N)

    for i, lab in enumerate(labels):
        plt.figure(figsize=(16,6))
        # actual vs mean
        plt.plot(x, trues[:N, i], 'k',      label='Actual')
        plt.plot(x, preds[:N, i], 'r--', label='Predicted')

        # fill the UQ band if provided
        if lower is not None and upper is not None:
            plt.fill_between(
                x,
                lower[:N, i],
                upper[:N, i],
                alpha=0.3,
                color='r',
                label='95% CI'
            )

        plt.xlabel('Time (minutes)')
        plt.ylabel(lab)
        plt.xlim(0,10_000)
        plt.grid()
        plt.title(f"{title} — {lab}")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


# -----------------------------------------------------------------------------
# 4) Load & preprocess full dataset
# -----------------------------------------------------------------------------
df = pd.read_csv('../../dataset/PSML.csv', parse_dates=['time'])
df.set_index('time', inplace=True)
df1 = df.fillna(method='ffill').fillna(method='bfill')

# features & targets
X, y = feature_label_split(df1,
    targets=['solar_power','wind_power'],
    drop_cols=['load_power']
)

# scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_arr = scaler_X.fit_transform(X)
y_arr = scaler_y.fit_transform(y)

# build sequences
seq_len = 12
X_seq, y_seq = create_sequences(
    torch.tensor(X_arr, dtype=torch.float32),
    torch.tensor(y_arr, dtype=torch.float32),
    seq_len
)

# full sequence dataset
train_ds = TensorDataset(X_seq, y_seq)
n_samples = len(train_ds)
print(f"Total sequence samples: {n_samples}")

# -----------------------------------------------------------------------------
# 5) Session window definitions
# -----------------------------------------------------------------------------
train_window = 43_800    # first session training size,1 month
test_window  = 43_800     # each session test size, 1 month

# hyperparameters
in_f    = X_seq.size(-1)
out_f   = y_seq.size(-1)
hidden  = 35
n_layers= 1
lr      = 1e-3
epochs  = 50
batch_size = 512
loss_fn = nn.MSELoss()
samples = 100

# metrics storage
session_nums      = []
training_time     = []
inference_time    = []
r2_solar_list     = []
r2_wind_list      = []
mae_solar_list    = []
mae_wind_list     = []
rmse_solar_list   = []
rmse_wind_list    = []
r2_list      = []
mae_list     = []
rmse_list    = []

# -----------------------------------------------------------------------------
# 6) Rolling train/test sessions (probabilistic LSTM)
# -----------------------------------------------------------------------------
session      = 0
train_start  = 0
train_end    = train_window
test_start   = train_end
test_end     = test_start + test_window

while test_end <= n_samples:
    print(f"\n=== Session {session} ===")
    # define indices
    train_idx = list(range(train_start, train_end))
    test_idx  = list(range(test_start, test_end))

    # loaders
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        Subset(train_ds, test_idx),
        batch_size=batch_size, shuffle=False, drop_last=True
    )

    # model setup
    if session == 0:
        model = LSTMReparameterizationModel(
            in_features=in_f,
            hidden_size=hidden,
            out_features=out_f,
            num_layers=n_layers,
            prior_mean=0,
            prior_variance=0.5,
            posterior_rho_init=-4.0,
            bias=True
        ).to(device)
    else:
        model.load_state_dict(
            torch.load(f"prob_lstm_session{session-1}.pth", map_location=device)
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train (assumes your train_lstm handles the KL term returned by the variational model)
    print(f" Training on samples [{train_start}:{train_end}]")
    start = time.time()
    train_model(model, train_loader, epochs, loss_fn, optimizer, device)
    end = time.time()
    train_time = end-start
    training_time.append(train_time)
    print(f"Training took {train_time:.2f}s")
    
    # test & store metrics
    print(f" Testing on samples [{test_start}:{test_end}]")
    start = time.time()
    mean_preds, trues, lower, upper = predict_with_uncertainty(
        model,
        test_loader,
        n_samples=samples,
        scaler_y=scaler_y,
        device=device,
        n_jobs=4,
        alpha=0.05
    )
    end = time.time()
    predict_time = end-start
    inference_time.append(predict_time)
    print(f"Inference took {predict_time:.2f}s")

    plot_predictions(
        preds=mean_preds,
        trues=trues,
        title=f"Session {session} Test Set",
        labels=['Solar','Wind'],
        n_display=test_window,
        lower=lower,
        upper=upper
    )

    # compute per-output metrics on the MEAN predictions
    r2_vals    = [r2_score(trues[:,i], mean_preds[:,i]) for i in range(out_f)]
    mae_vals   = [mean_absolute_error(trues[:,i], mean_preds[:,i]) for i in range(out_f)]
    rmse_vals  = [mean_squared_error(trues[:,i], mean_preds[:,i], squared=False) for i in range(out_f)]
    print(f"  R2   : {r2_vals}")
    print(f"  MAE  : {mae_vals}")
    print(f"  RMSE : {rmse_vals}")

    # append metrics
    session_nums.append(session)
    r2_solar_list .append(r2_vals[0])
    r2_wind_list  .append(r2_vals[1])
    mae_solar_list.append(mae_vals[0])
    mae_wind_list .append(mae_vals[1])
    rmse_solar_list.append(rmse_vals[0])
    rmse_wind_list .append(rmse_vals[1])
    r2_list.append(np.mean(r2_vals))
    mae_list.append(np.mean(mae_vals))
    rmse_list.append(np.mean(rmse_vals))

    # save probabilistic model state
    torch.save(model.state_dict(), f"prob_lstm_session{session}.pth")

    # advance windows
    train_start = test_start
    train_end   = test_end
    test_start  = train_end
    test_end    = test_start + test_window
    session    += 1

# -----------------------------------------------------------------------------
# 7) Plot metrics vs. session
# -----------------------------------------------------------------------------
def plot_metric(session_nums, solar_vals, wind_vals, ylabel, title):
    plt.figure(figsize=(8,4))
    plt.plot(session_nums, solar_vals, 'o-', label='Solar')
    plt.plot(session_nums, wind_vals,  'x--', label='Wind')
    plt.title(title)
    plt.xlabel('Session')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

plot_metric(session_nums, r2_solar_list,    r2_wind_list,    'R²',    'R² by Session & Output')
plot_metric(session_nums, mae_solar_list,   mae_wind_list,   'MAE',   'MAE by Session & Output')
plot_metric(session_nums, rmse_solar_list,  rmse_wind_list,  'RMSE',  'RMSE by Session & Output')

plt.figure(figsize=(8,4))
plt.plot(session_nums, r2_list, marker='o')
plt.title('Average R² vs. Session')
plt.xlabel('Session')
plt.ylabel('Avg R²')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(session_nums, mae_list, marker='o')
plt.title('Average MAE vs. Session')
plt.xlabel('Session')
plt.ylabel('Avg MAE')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.plot(session_nums, rmse_list, marker='o')
plt.title('Average RMSE vs. Session')
plt.xlabel('Session')
plt.ylabel('Avg RMSE')
plt.grid(True)
plt.show()

np.savez(
    "vlstm_rolling_metrics.npz",
    session_nums=np.array(session_nums),
    training_times=np.array(training_time),
    inference_times=np.array(inference_time),
    r2_solar=np.array(r2_solar_list),
    r2_wind=np.array(r2_wind_list),
    mae_solar=np.array(mae_solar_list),
    mae_wind=np.array(mae_wind_list),
    rmse_solar=np.array(rmse_solar_list),
    rmse_wind=np.array(rmse_wind_list),
)

print("Saved metrics.npz with all lists.")
