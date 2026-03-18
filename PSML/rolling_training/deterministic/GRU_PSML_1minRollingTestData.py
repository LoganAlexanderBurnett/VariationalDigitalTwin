import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from psml.data_handler import feature_label_split, create_sequences
from psml.models import RollingStandardGRUModel as StandardGRUModel

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
def train_gru(model, loader_train, epochs, loss_fn, optimizer, dev):
    model.to(dev)
    for ep in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader_train:
            xb, yb = xb.to(dev), yb.to(dev)
            optimizer.zero_grad()
            yp, _ = model(xb)
            loss = loss_fn(yp, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader_train)
        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep}/{epochs} — Train Loss: {avg_loss:.4f}")

def predict_standard_gru(model, loader, scaler_y, dev):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(dev)
            p, _ = model(xb)
            preds.append(p.cpu().numpy())
            trues.append(yb.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return scaler_y.inverse_transform(preds), scaler_y.inverse_transform(trues)

def plot_predictions(preds, trues, title, labels=['Solar','Wind'], n_display=500):
    N = min(len(preds), n_display)
    x = np.arange(N)
    for i, lab in enumerate(labels):
        plt.figure(figsize=(8,2.5))
        plt.plot(x, trues[:N,i], label='Actual')
        plt.plot(x, preds[:N,i], '--', label='Predicted')
        plt.title(f"{title} — {lab}")
        plt.legend(); plt.tight_layout(); plt.show()

def sym_mean_absolute_percentage_error(actual, predicted):
    return np.mean(
        np.abs(actual - predicted) / ((np.abs(actual) + np.abs(predicted)) / 2)
    ) * 100

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
train_window = 43_800    # first session training size, 1 month
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

# metrics storage
session_nums      = []
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
# 6) Rolling train/test sessions
# -----------------------------------------------------------------------------
session = 0
train_start = 0
train_end   = train_window
test_start  = train_end
test_end    = test_start + test_window

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
        model = StandardGRUModel(in_f, hidden, out_f, n_layers).to(device)
    else:
        model.load_state_dict(
            torch.load(f"gru_session{session-1}.pth", map_location=device)
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    print(f" Training on samples [{train_start}:{train_end}]")
    train_gru(model, train_loader, epochs, loss_fn, optimizer, device)

    # test & store metrics
    print(f" Testing on samples [{test_start}:{test_end}]")
    preds, trues = predict_standard_gru(model, test_loader, scaler_y, device)
    plot_predictions(preds, trues, title=f"Session {session} Test Set", n_display=test_window)

    # compute per-output metrics
    r2_vals   = [r2_score(trues[:,i], preds[:,i]) for i in range(out_f)]
    mae_vals  = [mean_absolute_error(trues[:,i], preds[:,i]) for i in range(out_f)]
    rmse_vals = [mean_squared_error(trues[:,i], preds[:,i], squared=False) for i in range(out_f)]
    smape_vals= [sym_mean_absolute_percentage_error(trues[:,i], preds[:,i]) for i in range(out_f)]
    print(f"  R2   : {r2_vals}")
    print(f"  MAE  : {mae_vals}")
    print(f"  RMSE : {rmse_vals}")
    print(f"  SMAPE: {smape_vals}")

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

    # save model
    torch.save(model.state_dict(), f"gru_session{session}.pth")

    # advance windows
    train_start = test_start
    train_end   = test_end
    test_start  = train_end
    test_end    = test_start + test_window
    session += 1

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
