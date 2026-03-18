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
    """
    Run MC sampling through a model to estimate predictive uncertainty.
    """
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            all_trues.append(targets.cpu().numpy())

            def sample_once():
                outs, _ = model(inputs)
                return outs.detach().cpu().numpy()

            batch_preds = Parallel(n_jobs=n_jobs)(
                delayed(sample_once)() for _ in range(n_samples)
            )
            all_preds.append(np.stack(batch_preds, axis=0))

    all_preds = np.concatenate(all_preds, axis=1)
    true_vals = np.concatenate(all_trues, axis=0)

    if scaler_y is not None:
        true_vals = scaler_y.inverse_transform(true_vals)
        all_preds = np.array([scaler_y.inverse_transform(p) for p in all_preds])

    mean_preds = np.mean(all_preds, axis=0)
    lower = np.percentile(all_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)

    return mean_preds, true_vals, lower, upper



def plot_predictions(preds, trues, n_display=10950, lower=None, upper=None):
    """
    Plot predictive means and optional confidence intervals.
    """
    labels = ['Solar', 'Wind']

    N = min(len(preds), n_display)
    x = np.arange(N)

    for i, lab in enumerate(labels):
        color = 'orangered' if lab == 'Solar' else 'dodgerblue'
        plt.figure(figsize=(16, 6))
        plt.plot(x, trues[:N, i], 'k', label='Actual')
        plt.plot(x, preds[:N, i], color=color, linestyle='--', label='Predicted')

        if lower is not None and upper is not None:
            plt.fill_between(
                x,
                lower[:N, i],
                upper[:N, i],
                alpha=0.5,
                color=color,
                label='95% CI',
            )

        plt.xlabel('Time (minutes)')
        plt.ylabel(lab)
        plt.grid()
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()



def calculate_and_display_metrics(true_values, mean_predictions):
    """
    Calculate and print R², MAE, and RMSE for each output.
    """
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
