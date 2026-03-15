from sklearn.model_selection import train_test_split
import torch


def feature_label_split(df, targets, drop_cols=[]):
    """
    Splits the dataframe into features and labels (targets).
    
    Parameters:
    df (DataFrame): The input dataframe.
    targets (list or string): The target column(s) to be used as labels. Can be a list of column names or a single column name.
    
    Returns:
    X (DataFrame): The features (input variables).
    y (DataFrame): The target labels (output variables).
    """
    # Ensure 'targets' is a list (even if a single string is passed)
    if isinstance(targets, str):
        targets = [targets]
    
    # Extract the target columns (y) and the remaining columns (X)
    y = df[targets]
    X = df.drop(columns=targets + drop_cols)
    
    return X, y


def train_val_test_split(X, y, train_fraction, validation_fraction, test_fraction):
    """
    Splits X, y into train, validation, and test sets according to provided fractions.
    Fractions must sum to 1. Uses sequential slicing (shuffle=False).
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    total = train_fraction + validation_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"train_fraction + validation_fraction + test_fraction must equal 1. Got {total:.6f}")
    
    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_fraction, shuffle=False
    )
    
    # Remaining fraction is (train + validation) = 1 - test_fraction
    remaining = train_fraction + validation_fraction  # which equals 1 - test_fraction
    
    # Compute validation ratio relative to the remaining data
    val_ratio = validation_fraction / remaining
    
    # Split X_temp/y_temp into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, shuffle=False
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def create_sequences(features, targets, seq_length):
    """
    Converts 2D features and targets into sequences of length `seq_length`.
    
    Args:
    - features: 2D tensor of shape [num_samples, num_features]
    - targets: 2D tensor of shape [num_samples, num_targets]
    - seq_length: The length of each sequence
    
    Returns:
    - seq_features: 3D tensor of shape [num_sequences, seq_length, num_features]
    - seq_targets: 2D tensor of shape [num_sequences, num_targets] (last target in each sequence)
    """
    seq_features = []
    seq_targets = []
    
    for i in range(len(features) - seq_length):
        seq_features.append(features[i:i+seq_length])  # Get sequences of features
        seq_targets.append(targets[i+seq_length])      # Get the target after the sequence
    
    return torch.stack(seq_features), torch.stack(seq_targets)
    