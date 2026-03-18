from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch


def predict_with_uncertainty(
    model,
    test_loader,
    n_samples=100,
    scaler_y=None,
    device=torch.device('cpu'),
    n_jobs=4,
    alpha=0.05,
):
    model.eval()
    all_predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            true_values.append(targets.cpu().numpy())

            def sample_prediction():
                outputs, _ = model(inputs)
                return outputs.detach().cpu().numpy()

            if n_jobs and n_jobs > 1:
                predictions = Parallel(n_jobs=n_jobs)(
                    delayed(sample_prediction)() for _ in range(n_samples)
                )
            else:
                predictions = [sample_prediction() for _ in range(n_samples)]

            predictions = np.stack(predictions, axis=0)
            all_predictions.append(predictions)

    all_predictions = np.concatenate(all_predictions, axis=1)
    true_values = np.concatenate(true_values, axis=0)

    if scaler_y is not None:
        true_values = scaler_y.inverse_transform(true_values)
        all_predictions = np.array(
            [scaler_y.inverse_transform(pred) for pred in all_predictions]
        )

    mean_predictions = np.mean(all_predictions, axis=0)
    lower = np.percentile(all_predictions, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_predictions, 100 * (1 - alpha / 2), axis=0)

    return mean_predictions, true_values, lower, upper


def calculate_mean_and_ci(predictions, confidence=0.95):
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    z = 1.96 if confidence == 0.95 else {0.90: 1.645, 0.99: 2.576}.get(confidence, 1.96)
    ci = z * std_predictions
    return mean_predictions, ci


def plot_predictions_with_ci(mean_predictions, ci, true_values, output_index=0):
    plt.figure(figsize=(16, 6))
    plt.plot(true_values[:, output_index], label='Actual', color='blue', linewidth=2)
    plt.plot(mean_predictions[:, output_index], label='Predicted Mean', color='red', linestyle='--', linewidth=2)
    plt.fill_between(
        np.arange(mean_predictions.shape[0]),
        mean_predictions[:, output_index] - ci[:, output_index],
        mean_predictions[:, output_index] + ci[:, output_index],
        color='red', alpha=0.2, label='95% Confidence Interval',
    )
    plt.title(f'Predicted vs Actual with 95% Confidence Interval (Output {output_index + 1})')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


def calculate_and_display_metrics(true_values, mean_predictions):
    n_outputs = true_values.shape[1]
    r2_scores = [r2_score(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    mae_scores = [mean_absolute_error(true_values[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    rmse_scores = [np.sqrt(mean_squared_error(true_values[:, i], mean_predictions[:, i])) for i in range(n_outputs)]

    for i in range(n_outputs):
        print(f"Output {i+1}:")
        print(f"  R² score: {r2_scores[i]:.4f}")
        print(f"  MAE score: {mae_scores[i]:.4f}")
        print(f"  RMSE score: {rmse_scores[i]:.4f}")
        print()
