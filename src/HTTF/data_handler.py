from sklearn.model_selection import train_test_split
import numpy as np
import torch


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
