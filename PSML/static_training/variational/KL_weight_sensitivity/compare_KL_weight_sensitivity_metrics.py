from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_WEIGHTS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]


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

    df = pd.DataFrame(records)
    df = df.sort_values("kl_weight").reset_index(drop=True)
    return df


def save_metric_plot(df: pd.DataFrame, metric: str, output_dir: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(df["kl_weight"], df[metric], marker="o")
    plt.xscale("log")
    plt.xlabel("KL Weight (log scale)")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"{metric.replace('_', ' ').title()} vs KL Weight")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_vs_kl_weight.png")
    plt.close()


def save_group_plot(df: pd.DataFrame, metrics: list[str], output_dir: Path, filename: str, title: str):
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(9, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    for axis, metric in zip(axes, metrics):
        axis.plot(df["kl_weight"], df[metric], marker="o")
        axis.set_xscale("log")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.grid(True, which="both", linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("KL Weight (log scale)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)


def main():
    results_root = Path(__file__).resolve().parent
    output_dir = results_root / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_metrics(results_root)

    metrics_df.to_csv(output_dir / "all_metrics_comparison.csv", index=False)

    metric_columns = [
        "solar_r2",
        "solar_mae",
        "solar_rmse",
        "wind_r2",
        "wind_mae",
        "wind_rmse",
        "train_time_sec",
        "infer_time_sec",
        "coverage_solar",
        "coverage_wind",
    ]

    available_metric_columns = [column for column in metric_columns if column in metrics_df.columns]

    for metric in available_metric_columns:
        save_metric_plot(metrics_df, metric, output_dir)

    performance_metrics = [
        metric
        for metric in ["solar_r2", "wind_r2", "solar_mae", "wind_mae", "solar_rmse", "wind_rmse"]
        if metric in metrics_df.columns
    ]
    if performance_metrics:
        save_group_plot(
            metrics_df,
            performance_metrics,
            output_dir,
            filename="performance_metrics_vs_kl_weight.png",
            title="Performance Metrics Across KL Weights",
        )

    coverage_metrics = [metric for metric in ["coverage_solar", "coverage_wind"] if metric in metrics_df.columns]
    if coverage_metrics:
        save_group_plot(
            metrics_df,
            coverage_metrics,
            output_dir,
            filename="coverage_metrics_vs_kl_weight.png",
            title="Coverage Metrics Across KL Weights",
        )

    timing_metrics = [metric for metric in ["train_time_sec", "infer_time_sec"] if metric in metrics_df.columns]
    if timing_metrics:
        save_group_plot(
            metrics_df,
            timing_metrics,
            output_dir,
            filename="timing_metrics_vs_kl_weight.png",
            title="Timing Metrics Across KL Weights",
        )

    print(f"Saved comparison artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
