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


def train_val_test_split(X, y, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)  # Calculate the ratio of validation to training size
    
    # Step 1: Split the data into (train+val) and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    
    # Step 2: Split the (train+val) set into actual training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    
    return X_train, X_val, X_test, y_train, y_val, y_test  # Return all six sets


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
    