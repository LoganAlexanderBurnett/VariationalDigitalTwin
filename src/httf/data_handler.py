from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def feature_label_split(df, targets, drop_cols=None):
    if drop_cols is None:
        drop_cols = []
    if isinstance(targets, str):
        targets = [targets]

    y = df[targets]
    X = df.drop(columns=targets + drop_cols)
    return X, y


def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sequences(features, targets, seq_length):
    seq_features = []
    seq_targets = []

    for i in range(len(features) - seq_length):
        seq_features.append(features[i:i + seq_length])
        seq_targets.append(targets[i + seq_length])

    return torch.stack(seq_features), torch.stack(seq_targets)


def create_autoregressive_sequences(data, lookback=10):
    seq_features = []
    seq_targets = []

    for i in range(lookback, len(data)):
        seq_features.append(data[i - lookback:i])
        seq_targets.append(data[i])

    if isinstance(data, torch.Tensor):
        return torch.stack(seq_features), torch.stack(seq_targets)

    return np.array(seq_features), np.array(seq_targets)


def report_nan_rows(df, column_name, name):
    nan_rows = df[df[column_name].isna()]
    if not nan_rows.empty:
        print(f"Rows with NaN values in {name} dataset:")
        print(nan_rows)
    else:
        print(f"No NaN values in {name} dataset.")


def prepare_autoregressive_splits(
    train_df,
    test_df,
    valid_df,
    lookback=10,
    device=torch.device("cpu"),
    scaler=None,
    interpolate_columns=None,
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    valid_df = valid_df.copy()

    if interpolate_columns:
        for column in interpolate_columns:
            train_df[column] = train_df[column].interpolate()
            test_df[column] = test_df[column].interpolate()
            valid_df[column] = valid_df[column].interpolate()

    scaler = MinMaxScaler(feature_range=(0, 1)) if scaler is None else scaler
    train_data = scaler.fit_transform(train_df)
    test_data = scaler.transform(test_df)
    valid_data = scaler.transform(valid_df)

    Xtrain, Ytrain = create_autoregressive_sequences(train_data, lookback=lookback)
    Xtest, Ytest = create_autoregressive_sequences(test_data, lookback=lookback)
    Xvalid, Yvalid = create_autoregressive_sequences(valid_data, lookback=lookback)

    return {
        "scaler": scaler,
        "train": (
            torch.tensor(Xtrain, dtype=torch.float32).to(device),
            torch.tensor(Ytrain, dtype=torch.float32).to(device),
        ),
        "test": (
            torch.tensor(Xtest, dtype=torch.float32).to(device),
            torch.tensor(Ytest, dtype=torch.float32).to(device),
        ),
        "valid": (
            torch.tensor(Xvalid, dtype=torch.float32).to(device),
            torch.tensor(Yvalid, dtype=torch.float32).to(device),
        ),
    }


def build_tensor_dataloader(features, targets, batch_size, shuffle=False, drop_last=False):
    dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
