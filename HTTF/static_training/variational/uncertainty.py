import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


from joblib import Parallel, delayed

def predict_with_uncertainty(model, test_loader, n_samples=100, scaler_y=None, device=torch.device('cpu'), n_jobs=4):
    model.eval()
    
    all_predictions = []
    true_values = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Store true values (used for comparison)
            true_values.append(targets.cpu().numpy())
            
            # Function to generate a single sample prediction
            def sample_prediction():
                outputs, _ = model(inputs)
                return outputs.detach().cpu().numpy()  # Ensure detached tensor
            
            # Use joblib Parallel for parallel sampling
            predictions = Parallel(n_jobs=n_jobs)(delayed(sample_prediction)() for _ in range(n_samples))

            # Stack predictions across samples
            predictions = np.stack(predictions, axis=0)  # Shape: (n_samples, batch_size, num_outputs)
            all_predictions.append(predictions)

    # Concatenate predictions across all batches
    all_predictions = np.concatenate(all_predictions, axis=1)  # Shape: (n_samples, total_samples, num_outputs)
    true_values = np.concatenate(true_values, axis=0)  # Shape: (total_samples, num_outputs)

    # Apply inverse scaling if scaler_y is provided
    if scaler_y is not None:
        true_values = scaler_y.inverse_transform(true_values)
        all_predictions = np.array([scaler_y.inverse_transform(pred) for pred in all_predictions])

    return all_predictions, true_values


# Function to make predictions multiple times to capture uncertainty
# def predict_with_uncertainty(model, test_loader, n_samples=100, scaler_y=None, device=torch.device('cpu')):
#     model.eval()
    
#     all_predictions = []
#     true_values = []
    
#     # Disable gradient computation for inference
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs = inputs.to(device)
#             targets = targets.to(device)
            
#             # Store true values (used for comparison)
#             true_values.append(targets.cpu().numpy())
            
#             # Generate multiple predictions for each input
#             predictions = []
#             for _ in range(n_samples):
#                 outputs, _ = model(inputs)
#                 predictions.append(outputs.cpu().numpy())
                
#             # Stack predictions across samples
#             predictions = np.stack(predictions, axis=0)  # Shape: (n_samples, batch_size, num_outputs)
#             all_predictions.append(predictions)

#     # Concatenate predictions across all batches
#     all_predictions = np.concatenate(all_predictions, axis=1)  # Shape: (n_samples, total_samples, num_outputs)
#     true_values = np.concatenate(true_values, axis=0)  # Shape: (total_samples, num_outputs)

#     # Apply inverse scaling if scaler_y is provided
#     if scaler_y is not None:
#         true_values = scaler_y.inverse_transform(true_values)
#         all_predictions = np.array([scaler_y.inverse_transform(pred) for pred in all_predictions])

#     return all_predictions, true_values


def calculate_mean_and_ci(predictions, confidence=0.95):
    # Get num samples - need to divide by num samples when computing confidence intervals -> CI = y_hat +/- z * (std / sqrt(n))
    n_samples = predictions.shape[0]

    # Mean of the predictions
    mean_predictions = np.mean(predictions, axis=0)  # Shape: (total_samples, num_outputs)

    # Std. deviation of predictions
    std_predictions = np.std(predictions, axis=0)    # Shape: (total_samples, num_outputs)

    # Adjust for sample size: divide std by sqrt(n)
    std_error = std_predictions # / np.sqrt(n_samples), giving wayyyy to small uncertainty windows

    # Z-score for the given confidence level (default is 1.96 for 95%)
    z = 1.96 if confidence == 0.95 else {
        0.90: 1.645,
        0.99: 2.576
    }.get(confidence, 1.96)
    
    # Confidence intervals: 1.96 * std for 95% confidence
    ci = z * std_error  # Shape: (total_samples, num_outputs)
    
    return mean_predictions, ci

# Plot the predictions along with confidence intervals
def plot_predictions_with_ci(mean_predictions, ci, true_values, output_index=0):
    """
    Visualize the predicted mean with confidence intervals and true values.
    
    Args:
    - mean_predictions: Mean of the predictions, shape (total_samples, num_outputs).
    - ci: Confidence interval values, shape (total_samples, num_outputs).
    - true_values: True target values, shape (total_samples, num_outputs).
    - output_index: The index of the output to visualize (0, 1, or 2).
    """
    plt.figure(figsize=(16, 6))
    
    # Plot true values (ground truth)
    plt.plot(true_values[:, output_index], label='Actual', color='blue', linewidth=2)

    # Plot predicted mean
    plt.plot(mean_predictions[:, output_index], label='Predicted Mean', color='red', linestyle='--', linewidth=2)
    
    # Plot confidence intervals (mean +/- confidence interval)
    plt.fill_between(
        np.arange(mean_predictions.shape[0]),
        mean_predictions[:, output_index] - ci[:, output_index],
        mean_predictions[:, output_index] + ci[:, output_index],
        color='red', alpha=0.2, label='95% Confidence Interval'
    )
    
    plt.title(f'Predicted vs Actual with 95% Confidence Interval (Output {output_index + 1})')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def calculate_and_display_metrics(true_values, mean_predictions):
    """
    Calculate R², MAE, and RMSE for each output and display the results.

    Parameters:
    - true_values: numpy array of true values, shape (n_samples, n_outputs)
    - mean_predictions: numpy array of predicted values, shape (n_samples, n_outputs)
    """
    # Number of outputs
    n_outputs = true_values.shape[1]
    
    # Calculate metrics for each output
    r2_scores = [r2_score(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    mae_scores = [mean_absolute_error(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    rmse_scores = [np.sqrt(mean_squared_error(true_values[:, i], mean_predictions[:, i])) for i in range(n_outputs)]
    
    # Display all the metrics for each output
    for i in range(n_outputs):
        print(f"Output {i+1}:")
        print(f"  R² score: {r2_scores[i]:.4f}")
        print(f"  MAE score: {mae_scores[i]:.4f}")
        print(f"  RMSE score: {rmse_scores[i]:.4f}")
        print()  # Add a blank line between outputs for clarity