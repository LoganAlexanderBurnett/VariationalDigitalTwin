from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from psml.data_handler import create_sequences, feature_label_split
from psml.models import LSTMReparameterizationModel
from psml.predict import plot_predictions, predict_with_uncertainty
from psml.trainer import set_random_seed, train_variational


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = '../../dataset/PSML.csv'
OUTPUT_ROOT = Path('vlstm_rolling_session_outputs')

TARGET_COLUMNS = ['solar_power', 'wind_power']
DROP_COLUMNS = ['load_power']

SEQ_LEN = 12
TRAIN_WINDOW = 43_800
TEST_WINDOW = 43_800

HIDDEN_SIZE = 35
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
EPOCHS = 50
BATCH_SIZE = 512

PRIOR_MEAN = 0.0
PRIOR_VARIANCE = 0.5
POSTERIOR_RHO_INIT = -4.0
BIAS = True

UNCERTAINTY_SAMPLES = 100
UNCERTAINTY_ALPHA = 0.05
UNCERTAINTY_N_JOBS = 4
LOG_EVERY = 10


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def sym_mean_absolute_percentage_error(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    denominator = np.where(denominator == 0, 1e-12, denominator)
    return np.mean(np.abs(actual - predicted) / denominator) * 100



def save_prediction_plot(mean_predictions, targets, lower, upper, labels, title, save_path, n_display):
    plot_predictions(
        preds=mean_predictions,
        trues=targets,
        title=title,
        labels=labels,
        n_display=min(n_display, len(mean_predictions)),
        lower=lower,
        upper=upper,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')



def save_session_predictions(mean_predictions, targets, lower, upper, target_names, save_path):
    df_out = pd.DataFrame()
    for i, name in enumerate(target_names):
        df_out[f'true_{name}'] = targets[:, i]
        df_out[f'pred_{name}'] = mean_predictions[:, i]
        df_out[f'lower_{name}'] = lower[:, i]
        df_out[f'upper_{name}'] = upper[:, i]
    df_out.to_csv(save_path, index=False)



def compute_metrics(mean_predictions, targets, target_names):
    n_outputs = targets.shape[1]

    r2_vals = [r2_score(targets[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    mae_vals = [mean_absolute_error(targets[:, i], mean_predictions[:, i]) for i in range(n_outputs)]
    rmse_vals = [np.sqrt(mean_squared_error(targets[:, i], mean_predictions[:, i])) for i in range(n_outputs)]
    smape_vals = [sym_mean_absolute_percentage_error(targets[:, i], mean_predictions[:, i]) for i in range(n_outputs)]

    metrics = {
        'r2': {target_names[i]: float(r2_vals[i]) for i in range(n_outputs)},
        'mae': {target_names[i]: float(mae_vals[i]) for i in range(n_outputs)},
        'rmse': {target_names[i]: float(rmse_vals[i]) for i in range(n_outputs)},
        'smape': {target_names[i]: float(smape_vals[i]) for i in range(n_outputs)},
        'avg_r2': float(np.mean(r2_vals)),
        'avg_mae': float(np.mean(mae_vals)),
        'avg_rmse': float(np.mean(rmse_vals)),
        'avg_smape': float(np.mean(smape_vals)),
    }
    return metrics



def save_metrics(metrics, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)



def plot_two_output_metric(session_nums, output_1_vals, output_2_vals, ylabel, title, label_1, label_2, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(session_nums, output_1_vals, 'o-', label=label_1)
    plt.plot(session_nums, output_2_vals, 'x--', label=label_2)
    plt.title(title)
    plt.xlabel('Session')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def plot_single_metric(session_nums, values, ylabel, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(session_nums, values, marker='o')
    plt.title(title)
    plt.xlabel('Session')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------
def main():
    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.MSELoss()

    # Load and preprocess data
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df.set_index('time', inplace=True)
    df = df.ffill().bfill()

    X, y = feature_label_split(
        df,
        targets=TARGET_COLUMNS,
        drop_cols=DROP_COLUMNS,
    )
    target_names = list(y.columns)
    target_labels = [name.replace('_', ' ').title() for name in target_names]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_arr = scaler_X.fit_transform(X)
    y_arr = scaler_y.fit_transform(y)

    X_seq, y_seq = create_sequences(
        torch.tensor(X_arr, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.float32),
        SEQ_LEN,
    )

    dataset = TensorDataset(X_seq, y_seq)
    n_samples = len(dataset)
    print(f'Total sequence samples: {n_samples}')

    input_size = X_seq.size(-1)
    output_size = y_seq.size(-1)

    session_nums = []
    r2_output_1 = []
    r2_output_2 = []
    mae_output_1 = []
    mae_output_2 = []
    rmse_output_1 = []
    rmse_output_2 = []
    avg_r2 = []
    avg_mae = []
    avg_rmse = []
    avg_smape = []
    training_times = []
    inference_times = []

    session = 0
    train_start = 0
    train_end = TRAIN_WINDOW
    test_start = train_end
    test_end = test_start + TEST_WINDOW

    while test_end <= n_samples:
        session_dir = OUTPUT_ROOT / f'session_{session:03d}'
        session_dir.mkdir(parents=True, exist_ok=True)

        print(f'\n=== Session {session} ===')
        print(f' Training on samples [{train_start}:{train_end}]')

        train_indices = list(range(train_start, train_end))
        test_indices = list(range(test_start, test_end))

        train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )
        test_loader = DataLoader(
            Subset(dataset, test_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
        )

        model = LSTMReparameterizationModel(
            in_features=input_size,
            hidden_size=HIDDEN_SIZE,
            out_features=output_size,
            num_layers=NUM_LAYERS,
            prior_mean=PRIOR_MEAN,
            prior_variance=PRIOR_VARIANCE,
            posterior_rho_init=POSTERIOR_RHO_INIT,
            bias=BIAS,
        ).to(device)
        if session > 0:
            previous_model_path = OUTPUT_ROOT / f'session_{session - 1:03d}' / 'vlstm_model.pth'
            model.load_state_dict(torch.load(previous_model_path, map_location=device))

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_start_time = time.perf_counter()
        train_variational(
            model,
            train_loader,
            optimizer,
            loss_fn,
            EPOCHS,
            device=device,
            log_every=LOG_EVERY,
            return_history=False,
        )
        train_end_time = time.perf_counter()
        training_time = train_end_time - train_start_time

        print(f' Testing on samples [{test_start}:{test_end}]')
        inference_start_time = time.perf_counter()
        mean_predictions, targets, lower, upper = predict_with_uncertainty(
            model,
            test_loader,
            n_samples=UNCERTAINTY_SAMPLES,
            scaler_y=scaler_y,
            device=device,
            n_jobs=UNCERTAINTY_N_JOBS,
            alpha=UNCERTAINTY_ALPHA,
        )
        inference_end_time = time.perf_counter()
        inference_time = inference_end_time - inference_start_time

        save_prediction_plot(
            mean_predictions=mean_predictions,
            targets=targets,
            lower=lower,
            upper=upper,
            labels=target_labels,
            title=f'Session {session} Test Set',
            save_path=session_dir / 'test_predictions.png',
            n_display=TEST_WINDOW,
        )

        metrics = compute_metrics(mean_predictions, targets, target_names)
        metrics.update({
            'session': session,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'training_time_seconds': float(training_time),
            'inference_time_seconds': float(inference_time),
        })

        print(f"  R2   : {[metrics['r2'][name] for name in target_names]}")
        print(f"  MAE  : {[metrics['mae'][name] for name in target_names]}")
        print(f"  RMSE : {[metrics['rmse'][name] for name in target_names]}")
        print(f"  SMAPE: {[metrics['smape'][name] for name in target_names]}")

        save_session_predictions(
            mean_predictions=mean_predictions,
            targets=targets,
            lower=lower,
            upper=upper,
            target_names=target_names,
            save_path=session_dir / 'predictions.csv',
        )
        save_metrics(metrics, session_dir / 'metrics.json')
        torch.save(model.state_dict(), session_dir / 'vlstm_model.pth')

        session_nums.append(session)
        r2_output_1.append(metrics['r2'][target_names[0]])
        r2_output_2.append(metrics['r2'][target_names[1]])
        mae_output_1.append(metrics['mae'][target_names[0]])
        mae_output_2.append(metrics['mae'][target_names[1]])
        rmse_output_1.append(metrics['rmse'][target_names[0]])
        rmse_output_2.append(metrics['rmse'][target_names[1]])
        avg_r2.append(metrics['avg_r2'])
        avg_mae.append(metrics['avg_mae'])
        avg_rmse.append(metrics['avg_rmse'])
        avg_smape.append(metrics['avg_smape'])
        training_times.append(training_time)
        inference_times.append(inference_time)

        train_start = test_start
        train_end = test_end
        test_start = train_end
        test_end = test_start + TEST_WINDOW
        session += 1

    plot_two_output_metric(
        session_nums,
        r2_output_1,
        r2_output_2,
        'R²',
        'R² by Session & Output',
        target_labels[0],
        target_labels[1],
        OUTPUT_ROOT / 'r2_by_session.png',
    )
    plot_two_output_metric(
        session_nums,
        mae_output_1,
        mae_output_2,
        'MAE',
        'MAE by Session & Output',
        target_labels[0],
        target_labels[1],
        OUTPUT_ROOT / 'mae_by_session.png',
    )
    plot_two_output_metric(
        session_nums,
        rmse_output_1,
        rmse_output_2,
        'RMSE',
        'RMSE by Session & Output',
        target_labels[0],
        target_labels[1],
        OUTPUT_ROOT / 'rmse_by_session.png',
    )

    plot_single_metric(
        session_nums,
        avg_r2,
        'Average R²',
        'Average R² vs. Session',
        OUTPUT_ROOT / 'avg_r2_by_session.png',
    )
    plot_single_metric(
        session_nums,
        avg_mae,
        'Average MAE',
        'Average MAE vs. Session',
        OUTPUT_ROOT / 'avg_mae_by_session.png',
    )
    plot_single_metric(
        session_nums,
        avg_rmse,
        'Average RMSE',
        'Average RMSE vs. Session',
        OUTPUT_ROOT / 'avg_rmse_by_session.png',
    )
    plot_single_metric(
        session_nums,
        avg_smape,
        'Average SMAPE',
        'Average SMAPE vs. Session',
        OUTPUT_ROOT / 'avg_smape_by_session.png',
    )

    summary_df = pd.DataFrame({
        'session': session_nums,
        f'r2_{target_names[0]}': r2_output_1,
        f'r2_{target_names[1]}': r2_output_2,
        f'mae_{target_names[0]}': mae_output_1,
        f'mae_{target_names[1]}': mae_output_2,
        f'rmse_{target_names[0]}': rmse_output_1,
        f'rmse_{target_names[1]}': rmse_output_2,
        'avg_r2': avg_r2,
        'avg_mae': avg_mae,
        'avg_rmse': avg_rmse,
        'avg_smape': avg_smape,
        'training_time_seconds': training_times,
        'inference_time_seconds': inference_times,
    })
    summary_df.to_csv(OUTPUT_ROOT / 'session_metrics_summary.csv', index=False)

    np.savez(
        OUTPUT_ROOT / 'vlstm_rolling_metrics.npz',
        session_nums=np.array(session_nums),
        training_times=np.array(training_times),
        inference_times=np.array(inference_times),
        r2_solar=np.array(r2_output_1),
        r2_wind=np.array(r2_output_2),
        mae_solar=np.array(mae_output_1),
        mae_wind=np.array(mae_output_2),
        rmse_solar=np.array(rmse_output_1),
        rmse_wind=np.array(rmse_output_2),
    )

    config = {
        'data_path': DATA_PATH,
        'output_root': str(OUTPUT_ROOT),
        'targets': TARGET_COLUMNS,
        'dropped_columns': DROP_COLUMNS,
        'seq_len': SEQ_LEN,
        'train_window': TRAIN_WINDOW,
        'test_window': TEST_WINDOW,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'prior_mean': PRIOR_MEAN,
        'prior_variance': PRIOR_VARIANCE,
        'posterior_rho_init': POSTERIOR_RHO_INIT,
        'bias': BIAS,
        'uncertainty_samples': UNCERTAINTY_SAMPLES,
        'uncertainty_alpha': UNCERTAINTY_ALPHA,
        'uncertainty_n_jobs': UNCERTAINTY_N_JOBS,
        'log_every': LOG_EVERY,
        'device': str(device),
        'total_sequence_samples': int(n_samples),
        'num_sessions': int(len(session_nums)),
    }
    save_metrics(config, OUTPUT_ROOT / 'run_config.json')


if __name__ == '__main__':
    main()
