"""
evaluate.py

Generate the final tables and figures for the Calgary temperature-forecasting
project without retraining any models.

Expected inputs
---------------
results/mlp/final_test_metrics.csv
results/mlp/final_test_predictions.csv       (optional, but recommended)
results/mlp/training_history.csv             (optional)

Outputs
-------
figures/final/
    rmse_vs_horizon.png
    mae_vs_horizon.png
    skill_vs_horizon.png
    bias_vs_horizon.png
    mlp_improvement_over_ols.png
    example_forecast_6h.png                  (if predictions file exists)
    validation_loss_by_horizon.png           (if training history exists)

results/final/
    final_comparison_table.csv
    mlp_vs_ols_improvement.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS_PATH = BASE_DIR / "results/mlp/final_test_metrics.csv"
DEFAULT_PREDICTIONS_PATH = BASE_DIR / "results/mlp/final_test_predictions.csv"
DEFAULT_HISTORY_PATH = BASE_DIR / "results/mlp/training_history.csv"

DEFAULT_FIGURE_DIR = BASE_DIR / "figures/final"
DEFAULT_TABLE_DIR = BASE_DIR / "results/final"

MODEL_ORDER = [
    "Persistence",
    "MultipleLinearRegression",
    "MLP",
]

MODEL_LABELS = {
    "Persistence": "Persistence",
    "MultipleLinearRegression": "Linear regression",
    "MLP": "MLP",
}

HORIZON_ORDER = [1, 6, 12, 24, 48]


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run "
            "'python train_mlp.py --evaluate-test' first."
        )

    df = pd.read_csv(path)

    required = {
        "horizon_hours",
        "split",
        "model",
        "n_samples",
        "mae_C",
        "rmse_C",
        "bias_C",
        "skill_vs_persistence",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Metrics file is missing columns: "
            + ", ".join(sorted(missing))
        )

    df = df[df["split"] == "test"].copy()

    if df.empty:
        raise ValueError("No rows with split='test' were found.")

    df["horizon_hours"] = pd.to_numeric(
        df["horizon_hours"],
        errors="raise",
    ).astype(int)

    unexpected_horizons = sorted(
        set(df["horizon_hours"]) - set(HORIZON_ORDER)
    )
    if unexpected_horizons:
        raise ValueError(
            f"Unexpected horizons in metrics file: {unexpected_horizons}"
        )

    unexpected_models = sorted(set(df["model"]) - set(MODEL_ORDER))
    if unexpected_models:
        raise ValueError(
            f"Unexpected model names in metrics file: {unexpected_models}"
        )

    expected_pairs = {
        (h, model)
        for h in HORIZON_ORDER
        for model in MODEL_ORDER
    }

    observed_pairs = set(
        zip(df["horizon_hours"], df["model"])
    )

    missing_pairs = expected_pairs - observed_pairs
    if missing_pairs:
        raise ValueError(
            f"Metrics file is missing horizon/model rows: "
            f"{sorted(missing_pairs)}"
        )

    return df


def save_comparison_tables(
    metrics: pd.DataFrame,
    table_dir: Path,
) -> pd.DataFrame:
    table_dir.mkdir(parents=True, exist_ok=True)

    comparison = (
        metrics.pivot(
            index="horizon_hours",
            columns="model",
            values=["mae_C", "rmse_C", "bias_C"],
        )
        .sort_index()
    )

    comparison.columns = [
        f"{metric}_{MODEL_LABELS[model].replace(' ', '_').lower()}"
        for metric, model in comparison.columns
    ]

    comparison = comparison.reset_index()
    comparison.to_csv(
        table_dir / "final_comparison_table.csv",
        index=False,
    )

    ols = (
        metrics[metrics["model"] == "MultipleLinearRegression"]
        .set_index("horizon_hours")
        .sort_index()
    )
    mlp = (
        metrics[metrics["model"] == "MLP"]
        .set_index("horizon_hours")
        .sort_index()
    )

    improvement = pd.DataFrame(
        {
            "horizon_hours": HORIZON_ORDER,
            "ols_rmse_C": ols.loc[HORIZON_ORDER, "rmse_C"].to_numpy(),
            "mlp_rmse_C": mlp.loc[HORIZON_ORDER, "rmse_C"].to_numpy(),
        }
    )

    improvement["mlp_rmse_reduction_vs_ols_C"] = (
        improvement["ols_rmse_C"] - improvement["mlp_rmse_C"]
    )

    improvement["mlp_rmse_reduction_vs_ols_percent"] = (
        100.0
        * improvement["mlp_rmse_reduction_vs_ols_C"]
        / improvement["ols_rmse_C"]
    )

    improvement.to_csv(
        table_dir / "mlp_vs_ols_improvement.csv",
        index=False,
    )

    return improvement


def metric_by_horizon_plot(
    metrics: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["o", "s", "^"]
    linestyles = ["--", "-.", "-"]

    for model, marker, linestyle in zip(
        MODEL_ORDER,
        markers,
        linestyles,
    ):
        sub = (
            metrics[metrics["model"] == model]
            .set_index("horizon_hours")
            .loc[HORIZON_ORDER]
        )

        ax.plot(
            HORIZON_ORDER,
            sub[metric_col].to_numpy(),
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=6,
            label=MODEL_LABELS[model],
        )

    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(HORIZON_ORDER)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def skill_plot(
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    markers = ["s", "^"]
    linestyles = ["-.", "-"]

    for model, marker, linestyle in zip(
        ["MultipleLinearRegression", "MLP"],
        markers,
        linestyles,
    ):
        sub = (
            metrics[metrics["model"] == model]
            .set_index("horizon_hours")
            .loc[HORIZON_ORDER]
        )

        ax.plot(
            HORIZON_ORDER,
            sub["skill_vs_persistence"].to_numpy(),
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            markersize=6,
            label=MODEL_LABELS[model],
        )

    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("Skill relative to persistence")
    ax.set_title("Forecast Skill Relative to Persistence")
    ax.set_xticks(HORIZON_ORDER)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def mlp_improvement_plot(
    improvement: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        improvement["horizon_hours"],
        improvement["mlp_rmse_reduction_vs_ols_percent"],
        marker="o",
        linewidth=1.8,
    )

    ax.axhline(0.0, linewidth=1.0, linestyle=":")
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MLP RMSE reduction relative to OLS (%)")
    ax.set_title("Additional Skill from Nonlinear Modelling")
    ax.set_xticks(HORIZON_ORDER)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def example_forecast_plot(
    predictions_path: Path,
    output_path: Path,
    horizon: int = 6,
    days: int = 7,
) -> None:
    if not predictions_path.exists():
        print(
            f"Skipping example forecast plot: "
            f"{predictions_path} does not exist."
        )
        return

    df = pd.read_csv(
        predictions_path,
        parse_dates=["timestamp", "valid_time"],
    )

    required = {
        "timestamp",
        "valid_time",
        "horizon_hours",
        "observed_temp_C",
        "persistence_pred_C",
        "linear_pred_C",
        "mlp_pred_C",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Predictions file is missing columns: "
            + ", ".join(sorted(missing))
        )

    sub = (
        df[df["horizon_hours"] == horizon]
        .sort_values("valid_time")
        .copy()
    )

    if sub.empty:
        print(
            f"Skipping example forecast plot: no {horizon} h predictions."
        )
        return

    start = sub["valid_time"].iloc[0]
    end = start + pd.Timedelta(days=days)
    sub = sub[sub["valid_time"] < end]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        sub["valid_time"],
        sub["observed_temp_C"],
        linewidth=2.0,
        label="Observed",
    )
    ax.plot(
        sub["valid_time"],
        sub["persistence_pred_C"],
        linestyle="--",
        linewidth=1.4,
        label="Persistence",
    )
    ax.plot(
        sub["valid_time"],
        sub["linear_pred_C"],
        linestyle="-.",
        linewidth=1.4,
        label="Linear regression",
    )
    ax.plot(
        sub["valid_time"],
        sub["mlp_pred_C"],
        linestyle=":",
        linewidth=1.8,
        label="MLP",
    )

    ax.set_xlabel("Valid time")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Example {horizon}-Hour Temperature Forecasts"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def training_history_plot(
    history_path: Path,
    output_path: Path,
) -> None:
    if not history_path.exists():
        print(
            f"Skipping training-history plot: "
            f"{history_path} does not exist."
        )
        return

    df = pd.read_csv(history_path)

    required = {
        "horizon_hours",
        "epoch",
        "validation_mse_standardized",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Training-history file is missing columns: "
            + ", ".join(sorted(missing))
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    for horizon in HORIZON_ORDER:
        sub = df[df["horizon_hours"] == horizon]

        if sub.empty:
            continue

        ax.plot(
            sub["epoch"],
            sub["validation_mse_standardized"],
            linewidth=1.5,
            label=f"{horizon} h",
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (standardized target)")
    ax.set_title("Validation Loss by Forecast Horizon")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Horizon")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def print_summary(
    metrics: pd.DataFrame,
    improvement: pd.DataFrame,
) -> None:
    rmse_table = (
        metrics.pivot(
            index="horizon_hours",
            columns="model",
            values="rmse_C",
        )
        .loc[HORIZON_ORDER, MODEL_ORDER]
        .rename(columns=MODEL_LABELS)
    )

    print("\nFinal 2025 test RMSE (°C)")
    print("-------------------------")
    print(rmse_table.round(3).to_string())

    print("\nMLP improvement over OLS")
    print("------------------------")
    print(
        improvement[
            [
                "horizon_hours",
                "mlp_rmse_reduction_vs_ols_C",
                "mlp_rmse_reduction_vs_ols_percent",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final tables and figures from saved test results."
        )
    )

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=DEFAULT_TABLE_DIR,
    )
    parser.add_argument(
        "--example-horizon",
        type=int,
        choices=HORIZON_ORDER,
        default=6,
    )
    parser.add_argument(
        "--example-days",
        type=int,
        default=7,
    )

    args = parser.parse_args()

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.metrics)

    improvement = save_comparison_tables(
        metrics,
        args.table_dir,
    )

    metric_by_horizon_plot(
        metrics,
        metric_col="rmse_C",
        ylabel="RMSE (°C)",
        title="Temperature Forecast RMSE vs Forecast Horizon",
        output_path=args.figure_dir / "rmse_vs_horizon.png",
    )

    metric_by_horizon_plot(
        metrics,
        metric_col="mae_C",
        ylabel="MAE (°C)",
        title="Temperature Forecast MAE vs Forecast Horizon",
        output_path=args.figure_dir / "mae_vs_horizon.png",
    )

    metric_by_horizon_plot(
        metrics,
        metric_col="bias_C",
        ylabel="Bias (°C)",
        title="Temperature Forecast Bias vs Forecast Horizon",
        output_path=args.figure_dir / "bias_vs_horizon.png",
    )

    skill_plot(
        metrics,
        args.figure_dir / "skill_vs_horizon.png",
    )

    mlp_improvement_plot(
        improvement,
        args.figure_dir / "mlp_improvement_over_ols.png",
    )

    example_forecast_plot(
        predictions_path=args.predictions,
        output_path=(
            args.figure_dir
            / f"example_forecast_{args.example_horizon}h.png"
        ),
        horizon=args.example_horizon,
        days=args.example_days,
    )

    training_history_plot(
        history_path=args.history,
        output_path=args.figure_dir / "validation_loss_by_horizon.png",
    )

    print_summary(metrics, improvement)

    print("\nSaved figures in:")
    print(f"  {args.figure_dir}")

    print("\nSaved tables in:")
    print(f"  {args.table_dir}")


if __name__ == "__main__":
    main()
