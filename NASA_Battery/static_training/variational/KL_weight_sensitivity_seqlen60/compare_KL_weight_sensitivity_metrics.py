from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


EXPECTED_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _compute_uncertainty_metrics(predictions_csv: Path) -> dict:
    df = pd.read_csv(predictions_csv)

    true_col = _first_present_column(df, ["True Voltage", "true", "y_true", "target", "voltage_true"])
    pred_col = _first_present_column(df, ["Predicted Mean Voltage", "pred_mean", "mean", "y_pred", "voltage_pred"])
    lower_col = _first_present_column(df, ["Lower CI Voltage", "lower", "lower_ci", "ci_lower"])
    upper_col = _first_present_column(df, ["Upper CI Voltage", "upper", "upper_ci", "ci_upper"])

    if not all([true_col, pred_col, lower_col, upper_col]):
        return {
            "coverage": float("nan"),
            "spearman_corr_uncertainty_error": float("nan"),
        }

    abs_error = (df[true_col] - df[pred_col]).abs()
    interval_width = df[upper_col] - df[lower_col]
    coverage = ((df[true_col] >= df[lower_col]) & (df[true_col] <= df[upper_col])).mean()
    spearman_corr = interval_width.corr(abs_error, method="spearman")

    return {
        "coverage": float(coverage),
        "spearman_corr_uncertainty_error": float(spearman_corr),
    }


def _find_predictions_csv(case_dir: Path) -> Path | None:
    preferred = [
        case_dir / "inference_predictions.csv",
        case_dir / "vBattTest.csv",
        case_dir / "predictions.csv",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    for candidate in sorted(case_dir.glob("*.csv")):
        if candidate.name != "metrics.csv":
            return candidate

    return None


def load_metrics(results_root: Path) -> pd.DataFrame:
    records = []
    missing_cases = []

    for kl_weight in EXPECTED_WEIGHTS:
        case_dir = results_root / f"KL_{kl_weight:.0e}"
        metrics_path = case_dir / "metrics.csv"

        if not metrics_path.exists():
            missing_cases.append(str(metrics_path))
            continue

        case_metrics = pd.read_csv(metrics_path)
        if case_metrics.empty:
            missing_cases.append(f"{metrics_path} (empty file)")
            continue

        metric_row = case_metrics.iloc[0].to_dict()

        predictions_csv = _find_predictions_csv(case_dir)
        if predictions_csv is not None:
            metric_row.update(_compute_uncertainty_metrics(predictions_csv))
        else:
            metric_row.setdefault("coverage", float("nan"))
            metric_row.setdefault("spearman_corr_uncertainty_error", float("nan"))

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
            "No metrics.csv files were found for expected KL cases. "
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
    with PdfPages(output_pdf) as pdf:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

        _plot_metric(axes[0, 0], df, "mae", "MAE vs KL Weight", "MAE")
        _plot_metric(axes[0, 1], df, "mse", "MSE vs KL Weight", "MSE")
        _plot_metric(axes[1, 0], df, "coverage", "Uncertainty Coverage vs KL Weight", "Coverage")
        _plot_metric(
            axes[1, 1],
            df,
            "spearman_corr_uncertainty_error",
            "Uncertainty-Error Spearman Corr vs KL Weight",
            "Spearman Correlation",
        )

        fig.suptitle("NASA Battery Variational BattNN KL Weight Sensitivity", fontsize=14)
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
