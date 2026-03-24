from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from psml.data_handler import create_sequences, feature_label_split
from psml.models import GRUReparameterizationModel
from psml.predict import predict_with_uncertainty
from psml.trainer import set_random_seed


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = '../../../dataset/PSML.csv'
OUTPUT_ROOT = Path(__file__).resolve().parent
KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

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

UNCERTAINTY_SAMPLES = 50
UNCERTAINTY_ALPHA = 0.05
UNCERTAINTY_N_JOBS = 4


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def train_variational_fixed_kl(
    model,
    train_loader,
    optimizer,
    reconstruction_loss_fn,
    num_epochs,
    kl_weight,
    device,
):
    model.to(device)
    for _ in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()


def compute_metrics(true_values, mean_predictions):
    return {
        "solar_r2": r2_score(true_values[:, 0], mean_predictions[:, 0]),
        "solar_mae": mean_absolute_error(true_values[:, 0], mean_predictions[:, 0]),
        "solar_rmse": np.sqrt(mean_squared_error(true_values[:, 0], mean_predictions[:, 0])),
        "wind_r2": r2_score(true_values[:, 1], mean_predictions[:, 1]),
        "wind_mae": mean_absolute_error(true_values[:, 1], mean_predictions[:, 1]),
        "wind_rmse": np.sqrt(mean_squared_error(true_values[:, 1], mean_predictions[:, 1])),
        "avg_r2": np.mean([
            r2_score(true_values[:, 0], mean_predictions[:, 0]),
            r2_score(true_values[:, 1], mean_predictions[:, 1]),
        ]),
        "avg_mae": np.mean([
            mean_absolute_error(true_values[:, 0], mean_predictions[:, 0]),
            mean_absolute_error(true_values[:, 1], mean_predictions[:, 1]),
        ]),
        "avg_rmse": np.mean([
            np.sqrt(mean_squared_error(true_values[:, 0], mean_predictions[:, 0])),
            np.sqrt(mean_squared_error(true_values[:, 1], mean_predictions[:, 1])),
        ]),
    }


def save_case_predictions(true_values, mean_predictions, lower_ci, upper_ci, filename: Path):
    pd.DataFrame(
        {
            'True Solar': true_values[:, 0],
            'True Wind': true_values[:, 1],
            'Predicted Mean Solar': mean_predictions[:, 0],
            'Predicted Mean Wind': mean_predictions[:, 1],
            'Lower CI Solar': lower_ci[:, 0],
            'Lower CI Wind': lower_ci[:, 1],
            'Upper CI Solar': upper_ci[:, 0],
            'Upper CI Wind': upper_ci[:, 1],
        }
    ).to_csv(filename, index=False)


def save_json(data, filename: Path):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    set_random_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Load and preprocess once
    df = pd.read_csv(DATA_PATH, parse_dates=['time'])
    df.set_index('time', inplace=True)
    df = df.ffill().bfill()

    X, y = feature_label_split(df, targets=TARGET_COLUMNS, drop_cols=DROP_COLUMNS)

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
    input_size = X_seq.size(-1)
    output_size = y_seq.size(-1)

    print(f'Total sequence samples: {n_samples}')

    loss_fn = nn.MSELoss()

    for kl_weight in KL_WEIGHTS:
        weight_label = f"KL_{kl_weight:.0e}"
        case_dir = OUTPUT_ROOT / weight_label
        case_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n===== Running rolling evaluation for {weight_label} =====")

        session = 0
        train_start = 0
        train_end = TRAIN_WINDOW
        test_start = train_end
        test_end = test_start + TEST_WINDOW

        all_true_values = []
        all_mean_predictions = []
        all_lower_ci = []
        all_upper_ci = []
        training_times = []
        inference_times = []
        session_rows = []

        while test_end <= n_samples:
            session_dir = case_dir / f'session_{session:03d}'
            session_dir.mkdir(parents=True, exist_ok=True)

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

            model = GRUReparameterizationModel(
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
                previous_model_path = case_dir / f'session_{session - 1:03d}' / 'vgru_model.pth'
                model.load_state_dict(torch.load(previous_model_path, map_location=device))

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            t0 = time.perf_counter()
            train_variational_fixed_kl(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                reconstruction_loss_fn=loss_fn,
                num_epochs=EPOCHS,
                kl_weight=kl_weight,
                device=device,
            )
            training_time = time.perf_counter() - t0

            t1 = time.perf_counter()
            mean_predictions, true_values, lower_ci, upper_ci = predict_with_uncertainty(
                model,
                test_loader,
                n_samples=UNCERTAINTY_SAMPLES,
                scaler_y=scaler_y,
                device=device,
                n_jobs=UNCERTAINTY_N_JOBS,
                alpha=UNCERTAINTY_ALPHA,
            )
            inference_time = time.perf_counter() - t1

            all_true_values.append(true_values)
            all_mean_predictions.append(mean_predictions)
            all_lower_ci.append(lower_ci)
            all_upper_ci.append(upper_ci)
            training_times.append(training_time)
            inference_times.append(inference_time)

            session_metrics = compute_metrics(true_values, mean_predictions)
            session_metrics.update(
                {
                    "session": session,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "training_time_seconds": training_time,
                    "inference_time_seconds": inference_time,
                }
            )
            session_rows.append(session_metrics)
            save_json(session_metrics, session_dir / 'metrics.json')

            save_case_predictions(
                true_values,
                mean_predictions,
                lower_ci,
                upper_ci,
                session_dir / 'vGRUTest.csv',
            )
            torch.save(model.state_dict(), session_dir / 'vgru_model.pth')

            print(
                f"{weight_label} | session {session:03d} | "
                f"R2(solar,wind)=({session_metrics['solar_r2']:.4f},{session_metrics['wind_r2']:.4f})"
            )

            train_start = test_start
            train_end = test_end
            test_start = train_end
            test_end = test_start + TEST_WINDOW
            session += 1

        true_values_all = np.concatenate(all_true_values, axis=0)
        mean_predictions_all = np.concatenate(all_mean_predictions, axis=0)
        lower_ci_all = np.concatenate(all_lower_ci, axis=0)
        upper_ci_all = np.concatenate(all_upper_ci, axis=0)

        coverage = ((true_values_all >= lower_ci_all) & (true_values_all <= upper_ci_all)).mean(axis=0)

        case_metrics = compute_metrics(true_values_all, mean_predictions_all)
        case_metrics.update(
            {
                "kl_weight": kl_weight,
                "coverage_solar": float(coverage[0]),
                "coverage_wind": float(coverage[1]),
                "train_time_sec": float(np.sum(training_times)),
                "infer_time_sec": float(np.sum(inference_times)),
                "avg_train_time_sec_per_session": float(np.mean(training_times)),
                "avg_infer_time_sec_per_session": float(np.mean(inference_times)),
                "num_sessions": int(len(training_times)),
            }
        )

        pd.DataFrame(session_rows).to_csv(case_dir / 'session_metrics_summary.csv', index=False)
        pd.DataFrame([case_metrics]).to_csv(case_dir / 'metrics.csv', index=False)
        save_case_predictions(
            true_values_all,
            mean_predictions_all,
            lower_ci_all,
            upper_ci_all,
            case_dir / 'vGRUTest.csv',
        )

        run_config = {
            "data_path": DATA_PATH,
            "kl_weight": kl_weight,
            "seq_len": SEQ_LEN,
            "train_window": TRAIN_WINDOW,
            "test_window": TEST_WINDOW,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "uncertainty_samples": UNCERTAINTY_SAMPLES,
            "uncertainty_alpha": UNCERTAINTY_ALPHA,
            "uncertainty_n_jobs": UNCERTAINTY_N_JOBS,
            "device": str(device),
            "num_sequences": int(n_samples),
            "num_sessions": int(len(training_times)),
        }
        save_json(run_config, case_dir / 'run_config.json')

        print(f"Completed {weight_label}. Case metrics saved to: {case_dir / 'metrics.csv'}")

    print("Finished rolling KL weight sensitivity evaluation.")


if __name__ == '__main__':
    main()
