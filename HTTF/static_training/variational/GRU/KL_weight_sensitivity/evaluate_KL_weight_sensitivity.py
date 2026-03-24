from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "src" / "httf").exists())
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from httf import (
    VariationalGRUModel,
    build_tensor_dataloader,
    evaluate_variational_model,
    plot_loss_curves,
    plot_prediction_series,
    prepare_csv_autoregressive_splits,
    print_metrics_report,
    print_split_shapes,
    set_random_seed,
)


KL_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
DATA_DIR = Path(__file__).resolve().parent


def train_variational_fixed_kl(
    model,
    train_loader,
    optimizer,
    reconstruction_loss_fn,
    num_epochs,
    kl_weight,
    device,
    val_loader=None,
):
    model.to(device)
    history = {"train_losses": [], "val_losses": []}

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, kl_loss = model(inputs)
            reconstruction_loss = reconstruction_loss_fn(outputs, targets)
            total_loss = reconstruction_loss + kl_weight * kl_loss
            total_loss.backward()
            optimizer.step()
            running_train_loss += total_loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        history["train_losses"].append(avg_train_loss)

        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_targets in val_loader:
                    val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                    val_outputs, val_kl = model(val_inputs)
                    val_recon = reconstruction_loss_fn(val_outputs, val_targets)
                    running_val_loss += (val_recon + kl_weight * val_kl).item()

            avg_val_loss = running_val_loss / len(val_loader)
            history["val_losses"].append(avg_val_loss)
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] KL={kl_weight:.0e} "
                f"Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
            )

    return history


def save_case_predictions(true_values, mean_predictions, lower_ci, upper_ci, output_path: Path):
    df = pd.DataFrame(
        {
            "True Output 1": true_values[:, 0],
            "True Output 2": true_values[:, 1],
            "Predicted Mean Output 1": mean_predictions[:, 0],
            "Predicted Mean Output 2": mean_predictions[:, 1],
            "Lower CI Output 1": lower_ci[:, 0],
            "Lower CI Output 2": lower_ci[:, 1],
            "Upper CI Output 1": upper_ci[:, 0],
            "Upper CI Output 2": upper_ci[:, 1],
        }
    )
    df.to_csv(output_path, index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_random_seed()

    prepared = prepare_csv_autoregressive_splits(
        DATA_DIR,
        train_name="../../../../dataset/static_dataset/TF_TS_train.csv",
        test_name="../../../../dataset/static_dataset/TF_TS_test.csv",
        valid_name="../../../../dataset/static_dataset/TF_TS_valid.csv",
        lookback=10,
        device=device,
        interpolate_columns=["TS"],
    )
    print_split_shapes(prepared["train"], prepared["test"], prepared["valid"])

    scaler = prepared["scaler"]
    Xtrain_tensor, Ytrain_tensor = prepared["train"]
    Xtest_tensor, Ytest_tensor = prepared["test"]
    Xvalid_tensor, Yvalid_tensor = prepared["valid"]

    batch_size = 256
    num_epochs = 50

    train_loader = build_tensor_dataloader(Xtrain_tensor, Ytrain_tensor, batch_size=batch_size)
    valid_loader = build_tensor_dataloader(Xvalid_tensor, Yvalid_tensor, batch_size=batch_size)
    test_loader = build_tensor_dataloader(Xtest_tensor, Ytest_tensor, batch_size=batch_size)

    for kl_weight in KL_WEIGHTS:
        case_dir = DATA_DIR / f"KL_{kl_weight:.0e}"
        case_dir.mkdir(parents=True, exist_ok=True)

        model = VariationalGRUModel(
            in_features=2,
            hidden_size1=48,
            hidden_size2=64,
            hidden_size3=32,
            out_features=2,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00018)

        start = time.time()
        history = train_variational_fixed_kl(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            reconstruction_loss_fn=criterion,
            num_epochs=num_epochs,
            kl_weight=kl_weight,
            device=device,
            val_loader=valid_loader,
        )
        train_time = time.time() - start

        plot_loss_curves(history, case_dir / "training.png", yscale="log")

        start = time.time()
        evaluation = evaluate_variational_model(
            model,
            test_loader,
            scaler_y=scaler,
            device=device,
            output_dir=case_dir,
            n_samples=250,
            output_names=["output_1", "output_2"],
        )
        infer_time = time.time() - start

        print_metrics_report(evaluation["metrics"])

        plot_prediction_series(
            evaluation["true_values"][:, 0],
            evaluation["mean_predictions"][:, 0],
            case_dir / "VBBL_TS.png",
            ylabel="Solid Temperature (°C)",
            predicted_label="Variational GRU Mean",
            lower=evaluation["lower_ci"][:, 0],
            upper=evaluation["upper_ci"][:, 0],
            color="darkcyan",
            test_length=prepared["test_length"],
        )
        plot_prediction_series(
            evaluation["true_values"][:, 1],
            evaluation["mean_predictions"][:, 1],
            case_dir / "VBBL_TF.png",
            ylabel="Fluid Temperature (°C)",
            predicted_label="Variational GRU Mean",
            lower=evaluation["lower_ci"][:, 1],
            upper=evaluation["upper_ci"][:, 1],
            color="darkcyan",
            test_length=prepared["test_length"],
        )

        coverage = (
            (evaluation["true_values"] >= evaluation["lower_ci"]) &
            (evaluation["true_values"] <= evaluation["upper_ci"])
        ).mean(axis=0)

        metrics = {
            "kl_weight": kl_weight,
            "output_1_r2": evaluation["metrics"]["output_1"]["r2"],
            "output_1_mae": evaluation["metrics"]["output_1"]["mae"],
            "output_1_rmse": evaluation["metrics"]["output_1"]["rmse"],
            "output_2_r2": evaluation["metrics"]["output_2"]["r2"],
            "output_2_mae": evaluation["metrics"]["output_2"]["mae"],
            "output_2_rmse": evaluation["metrics"]["output_2"]["rmse"],
            "coverage_output_1": float(coverage[0]),
            "coverage_output_2": float(coverage[1]),
            "train_time_sec": float(train_time),
            "infer_time_sec": float(infer_time),
        }
        pd.DataFrame([metrics]).to_csv(case_dir / "metrics.csv", index=False)

        save_case_predictions(
            evaluation["true_values"],
            evaluation["mean_predictions"],
            evaluation["lower_ci"],
            evaluation["upper_ci"],
            case_dir / "vGRUTest.csv",
        )

        print(f"Completed KL={kl_weight:.0e}. Outputs saved to: {case_dir}")


if __name__ == "__main__":
    main()
