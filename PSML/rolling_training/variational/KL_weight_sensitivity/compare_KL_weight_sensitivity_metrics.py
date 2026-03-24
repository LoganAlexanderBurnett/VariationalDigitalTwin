from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


EXPECTED_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


def _compute_uncertainty_error_spearman(vgru_test_csv: Path) -> tuple[float, float]:
    df = pd.read_csv(vgru_test_csv)

    solar_abs_error = (df["True Solar"] - df["Predicted Mean Solar"]).abs()
    wind_abs_error = (df["True Wind"] - df["Predicted Mean Wind"]).abs()

    solar_interval_width = df["Upper CI Solar"] - df["Lower CI Solar"]
    wind_interval_width = df["Upper CI Wind"] - df["Lower CI Wind"]

    solar_spearman = solar_interval_width.corr(solar_abs_error, method="spearman")
    wind_spearman = wind_interval_width.corr(wind_abs_error, method="spearman")
    return solar_spearman, wind_spearman


def _load_session_metrics(case_dir: Path, kl_weight: float) -> pd.DataFrame:
    session_metrics_path = case_dir / "session_metrics_summary.csv"
    if not session_metrics_path.exists():
        return pd.DataFrame()

    session_df = pd.read_csv(session_metrics_path)
    if session_df.empty:
        return pd.DataFrame()

    session_df["kl_weight"] = kl_weight
    session_df["case_dir"] = case_dir.name
    return session_df


def load_metrics(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    case_records = []
    session_frames = []
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
            solar_spearman, wind_spearman = _compute_uncertainty_error_spearman(vgru_test_path)
            metric_row["spearman_corr_uncertainty_error_solar"] = solar_spearman
            metric_row["spearman_corr_uncertainty_error_wind"] = wind_spearman
        else:
            metric_row["spearman_corr_uncertainty_error_solar"] = float("nan")
            metric_row["spearman_corr_uncertainty_error_wind"] = float("nan")

        metric_row["case_dir"] = case_dir.name
        metric_row["metrics_path"] = str(metrics_path)

        if "kl_weight" not in metric_row:
            metric_row["kl_weight"] = kl_weight

        case_records.append(metric_row)

        session_df = _load_session_metrics(case_dir, kl_weight)
        if not session_df.empty:
            session_frames.append(session_df)

    if missing_cases:
        print("Missing or empty metrics files:")
        for missing_case in missing_cases:
            print(f"  - {missing_case}")

    if not case_records:
        raise FileNotFoundError(
            "No metrics.csv files were found for the expected KL cases. "
            "Run evaluate_KL_weight_sensitivity.py first."
        )

    case_df = pd.DataFrame(case_records).sort_values("kl_weight").reset_index(drop=True)

    if session_frames:
        session_df = pd.concat(session_frames, ignore_index=True)
        session_df = session_df.sort_values(["session", "kl_weight"]).reset_index(drop=True)
    else:
        session_df = pd.DataFrame()

    return case_df, session_df


def _plot_metric_vs_kl(ax, df: pd.DataFrame, y_col: str, title: str, ylabel: str):
    if y_col in df.columns:
        ax.plot(df["kl_weight"], df[y_col], marker="o")
    else:
        ax.text(0.5, 0.5, f"Missing metric: {y_col}", ha="center", va="center", transform=ax.transAxes)

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("KL Weight (log scale)")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)


def _plot_metric_vs_session(ax, session_df: pd.DataFrame, y_col: str, title: str, ylabel: str):
    if y_col not in session_df.columns:
        ax.text(0.5, 0.5, f"Missing metric: {y_col}", ha="center", va="center", transform=ax.transAxes)
        return

    for kl_weight in sorted(session_df["kl_weight"].unique()):
        subset = session_df[session_df["kl_weight"] == kl_weight].sort_values("session")
        ax.plot(subset["session"], subset[y_col], marker="o", label=f"KL={kl_weight:.0e}")

    ax.set_title(title)
    ax.set_xlabel("Session")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)


def save_session_behavior_pdf(session_df: pd.DataFrame, output_pdf: Path):
    if session_df.empty:
        print("No session_metrics_summary.csv files found; skipping session-behavior PDF.")
        return

    pages = [
        {
            "output": "Solar",
            "r2": "solar_r2",
            "mae": "solar_mae",
            "rmse": "solar_rmse",
            "avg": "avg_rmse",
        },
        {
            "output": "Wind",
            "r2": "wind_r2",
            "mae": "wind_mae",
            "rmse": "wind_rmse",
            "avg": "avg_rmse",
        },
    ]

    with PdfPages(output_pdf) as pdf:
        for page in pages:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

            _plot_metric_vs_session(axes[0, 0], session_df, page["r2"], f"{page['output']} R² by Session", "R²")
            _plot_metric_vs_session(axes[0, 1], session_df, page["mae"], f"{page['output']} MAE by Session", "MAE")
            _plot_metric_vs_session(axes[1, 0], session_df, page["rmse"], f"{page['output']} RMSE by Session", "RMSE")
            _plot_metric_vs_session(
                axes[1, 1],
                session_df,
                page["avg"],
                f"Reference Average Metric by Session ({page['output']})",
                page["avg"],
            )

            fig.suptitle(f"Session-wise Metric Behavior Across KL Weights — {page['output']}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def save_kl_summary_pdf(case_df: pd.DataFrame, output_pdf: Path):
    pages = [
        {
            "output": "Solar",
            "coverage": "coverage_solar",
            "r2": "solar_r2",
            "mae": "solar_mae",
            "rmse": "solar_rmse",
        },
        {
            "output": "Wind",
            "coverage": "coverage_wind",
            "r2": "wind_r2",
            "mae": "wind_mae",
            "rmse": "wind_rmse",
        },
    ]

    with PdfPages(output_pdf) as pdf:
        for page in pages:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

            _plot_metric_vs_kl(axes[0, 0], case_df, page["coverage"], f"{page['output']} Coverage vs KL Weight", "Coverage")
            _plot_metric_vs_kl(axes[0, 1], case_df, page["r2"], f"{page['output']} R² vs KL Weight", "R²")
            _plot_metric_vs_kl(axes[1, 0], case_df, page["mae"], f"{page['output']} MAE vs KL Weight", "MAE")
            _plot_metric_vs_kl(axes[1, 1], case_df, page["rmse"], f"{page['output']} RMSE vs KL Weight", "RMSE")

            fig.suptitle(f"Rolling KL Weight Sensitivity Metrics — {page['output']}", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)


def main():
    results_root = Path(__file__).resolve().parent
    output_dir = results_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    case_df, session_df = load_metrics(results_root)

    case_df.to_csv(output_dir / "all_metrics_comparison.csv", index=False)
    if not session_df.empty:
        session_df.to_csv(output_dir / "session_metrics_comparison.csv", index=False)

    session_pdf = output_dir / "session_behavior_across_kl.pdf"
    save_session_behavior_pdf(session_df, session_pdf)

    output_pdf = output_dir / "kl_weight_metrics_summary.pdf"
    save_kl_summary_pdf(case_df, output_pdf)

    print(f"Saved case-level comparison table to: {output_dir / 'all_metrics_comparison.csv'}")
    if not session_df.empty:
        print(f"Saved session-level comparison table to: {output_dir / 'session_metrics_comparison.csv'}")
    print(f"Saved session-behavior PDF to: {session_pdf}")
    print(f"Saved KL summary PDF to: {output_pdf}")


if __name__ == "__main__":
    main()
