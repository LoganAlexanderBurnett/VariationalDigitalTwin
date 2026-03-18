from sklearn.model_selection import train_test_split
import torch


def feature_label_split(df, targets, drop_cols=None):
    """
    Split a dataframe into features and targets.
    """
    if drop_cols is None:
        drop_cols = []
    if isinstance(targets, str):
        targets = [targets]

    y = df[targets]
    X = df.drop(columns=targets + drop_cols)
    return X, y



def train_val_test_split(X, y, *args, **kwargs):
    """
    Support both historical PSML split signatures:

    * train_val_test_split(X, y, test_ratio)
    * train_val_test_split(X, y, train_fraction, validation_fraction, test_fraction)
    """
    if "test_ratio" in kwargs:
        test_ratio = kwargs["test_ratio"]
        val_ratio = test_ratio / (1 - test_ratio)
    elif len(args) == 1 and not kwargs:
        test_ratio = args[0]
        val_ratio = test_ratio / (1 - test_ratio)
    elif len(args) == 3 and not kwargs:
        train_fraction, validation_fraction, test_fraction = args
        total = train_fraction + validation_fraction + test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError(
                "train_fraction + validation_fraction + test_fraction must equal 1. "
                f"Got {total:.6f}"
            )
        test_ratio = test_fraction
        remaining = train_fraction + validation_fraction
        val_ratio = validation_fraction / remaining
    else:
        raise TypeError(
            "train_val_test_split expects either (X, y, test_ratio) or "
            "(X, y, train_fraction, validation_fraction, test_fraction)"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, shuffle=False
    )
    return X_train, X_val, X_test, y_train, y_val, y_test



def create_sequences(features, targets, seq_length):
    """
    Convert 2D features and targets into rolling sequences.
    """
    seq_features = []
    seq_targets = []

    for i in range(len(features) - seq_length):
        seq_features.append(features[i : i + seq_length])
        seq_targets.append(targets[i + seq_length])

    return torch.stack(seq_features), torch.stack(seq_targets)
