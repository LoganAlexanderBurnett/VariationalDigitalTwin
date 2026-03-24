from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


EXPECTED_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _compute_uncertainty_metrics_from_vgrutest(vgru_test_csv: Path) -> dict:
    df = pd.read_csv(vgru_test_csv)

    output_1_abs_error = (df["True Output 1"] - df["Predicted Mean Output 1"]).abs()
    output_2_abs_error = (df["True Output 2"] - df["Predicted Mean Output 2"]).abs()

    output_1_interval_width = df["Upper CI Output 1"] - df["Lower CI Output 1"]
    output_2_interval_width = df["Upper CI Output 2"] - df["Lower CI Output 2"]

    output_1_coverage = (
        (df["True Output 1"] >= df["Lower CI Output 1"]) &
        (df["True Output 1"] <= df["Upper CI Output 1"])
    ).mean()
    output_2_coverage = (
        (df["True Output 2"] >= df["Lower CI Output 2"]) &
        (df["True Output 2"] <= df["Upper CI Output 2"])
    ).mean()

    output_1_spearman = output_1_interval_width.corr(output_1_abs_error, method="spearman")
    output_2_spearman = output_2_interval_width.corr(output_2_abs_error, method="spearman")

    return {
        "coverage_output_1": float(output_1_coverage),
        "coverage_output_2": float(output_2_coverage),
        "spearman_corr_uncertainty_error_output_1": float(output_1_spearman),
        "spearman_corr_uncertainty_error_output_2": float(output_2_spearman),
    }


def load_metrics(results_root: Path) -> pd.DataFrame:
    records = []
    missing_cases = []

    for kl_weight in EXPECTED_WEIGHTS:
        case_dir = results_root / f"KL_{kl_weight:.0e}"
        metrics_path = case_dir / "metrics.csv"
        vgru_test_path = case_dir / "vGRUTest.csv"

        if not metrics_path.exists():
            missing_cases.append(str(metrics_path))
            continue

        case_metrics = pd.read_csv(metrics_path)
        if case_metrics.empty:
            missing_cases.append(f"{metrics_path} (empty file)")
            continue

        metric_row = case_metrics.iloc[0].to_dict()

        if vgru_test_path.exists():
            metric_row.update(_compute_uncertainty_metrics_from_vgrutest(vgru_test_path))
        else:
            metric_row["coverage_output_1"] = float("nan")
            metric_row["coverage_output_2"] = float("nan")
            metric_row["spearman_corr_uncertainty_error_output_1"] = float("nan")
            metric_row["spearman_corr_uncertainty_error_output_2"] = float("nan")

        metric_row["case_dir"] = case_dir.name
        metric_row["metrics_path"] = str(metrics_path)

        if "kl_weight" not in metric_row:
            metric_row["kl_weight"] = kl_weight

        records.append(metric_row)

    if missing_cases:
        print("Missing or empty metrics files:")
        for missing_case in missing_cases:
            print(f"  - {missing_case}")

    if not records:
        raise FileNotFoundError(
            "No metrics.csv files were found for the expected KL cases. "
            "Run evaluate_KL_weight_sensitivity.py first."
        )

    return pd.DataFrame(records).sort_values("kl_weight").reset_index(drop=True)


def _plot_metric(ax, df: pd.DataFrame, y_col: str, title: str, ylabel: str):
    if y_col in df.columns:
        ax.plot(df["kl_weight"], df[y_col], marker="o")
    else:
        ax.text(0.5, 0.5, f"Missing metric: {y_col}", ha="center", va="center", transform=ax.transAxes)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("KL Weight (log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


def save_summary_pdf(df: pd.DataFrame, output_pdf: Path):
    pages = [
        {
            "output": "Output 1 (TS)",
            "coverage": "coverage_output_1",
            "r2": "output_1_r2",
            "mae": "output_1_mae",
            "rmse": "output_1_rmse",
        },
        {
            "output": "Output 2 (TF)",
            "coverage": "coverage_output_2",
            "r2": "output_2_r2",
            "mae": "output_2_mae",
            "rmse": "output_2_rmse",
        },
    ]

    with PdfPages(output_pdf) as pdf:
        for page in pages:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

            _plot_metric(axes[0, 0], df, page["coverage"], f"{page['output']} Coverage vs KL Weight", "Coverage")
            _plot_metric(axes[0, 1], df, page["r2"], f"{page['output']} R² vs KL Weight", "R²")
            _plot_metric(axes[1, 0], df, page["mae"], f"{page['output']} MAE vs KL Weight", "MAE")
            _plot_metric(axes[1, 1], df, page["rmse"], f"{page['output']} RMSE vs KL Weight", "RMSE")

            fig.suptitle(f"HTTF Variational GRU KL Weight Sensitivity — {page['output']}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def main():
    results_root = Path(__file__).resolve().parent
    output_dir = results_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(results_root)
    metrics_df.to_csv(output_dir / "all_metrics_comparison.csv", index=False)

    output_pdf = output_dir / "kl_weight_metrics_summary.pdf"
    save_summary_pdf(metrics_df, output_pdf)

    print(f"Saved comparison table to: {output_dir / 'all_metrics_comparison.csv'}")
    print(f"Saved summary plots PDF to: {output_pdf}")


if __name__ == "__main__":
    main()
