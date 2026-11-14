"""
train.py

Train and evaluate the baseline models for the Calgary temperature forecasting
project.

Default experimental split
--------------------------
Training:   all samples whose forecast origin AND target time are before 2024-01-01
Validation: forecast origin >= 2024-01-01 and target time < 2025-01-01
Test:       forecast origin >= 2025-01-01 and target time < 2026-01-01

The test set is NOT evaluated by default. Use --evaluate-test only after the
MLP architecture / training procedure has been fixed.

Models
------
1. Persistence
2. Multiple linear regression (OLS)

For each horizon h in {1, 6, 12, 24, 48}, this script:
    - creates horizon-safe chronological splits,
    - computes feature standardization statistics from training data only,
    - standardizes the OLS inputs,
    - fits OLS on training data,
    - evaluates persistence and OLS on validation data,
    - optionally evaluates on the held-out test data,
    - saves metrics, predictions, scaler statistics, and OLS coefficients.

Expected input
--------------
data/processed/model_ready.csv

This file should be produced by download_and_preprocess_weather.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models import (
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    MultipleLinearRegression,
    PersistenceModel,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/model_ready.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/baselines"

VAL_START = pd.Timestamp("2024-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }


def load_model_ready(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run download_and_preprocess_weather.py first."
        )

    df = pd.read_csv(path)

    required_cols = [
        "timestamp",
        *FEATURE_COLS,
        *TARGET_COLS.values(),
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            "model_ready.csv is missing required columns:\n"
            + "\n".join(missing)
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    df = df.sort_values("timestamp").reset_index(drop=True)

    numeric_cols = [*FEATURE_COLS, *TARGET_COLS.values()]
    if df[numeric_cols].isna().any().any():
        raise ValueError(
            "model_ready.csv contains missing predictor/target values. "
            "Re-run preprocessing or inspect the missingness report."
        )

    if not np.isfinite(df[numeric_cols].to_numpy(dtype=np.float64)).all():
        raise ValueError("model_ready.csv contains infinite numeric values.")

    return df


def horizon_safe_split(
    df: pd.DataFrame,
    horizon: int,
) -> dict[str, pd.DataFrame]:
    """
    Split by forecast origin while ensuring the target time stays in the
    same partition.

    Example:
        a 48 h training sample starting 2023-12-31 cannot be used because its
        target lies in the validation year.
    """
    origin = df["timestamp"]
    valid_time = origin + pd.to_timedelta(horizon, unit="h")

    train_mask = (
        (origin < VAL_START)
        & (valid_time < VAL_START)
    )

    val_mask = (
        (origin >= VAL_START)
        & (origin < TEST_START)
        & (valid_time < TEST_START)
    )

    test_mask = (
        (origin >= TEST_START)
        & (origin < TEST_END)
        & (valid_time < TEST_END)
    )

    splits = {
        "train": df.loc[train_mask].copy(),
        "validation": df.loc[val_mask].copy(),
        "test": df.loc[test_mask].copy(),
    }

    for name, split_df in splits.items():
        if split_df.empty:
            raise ValueError(
                f"The {name} split is empty for the {horizon} h horizon."
            )

    return splits


def fit_standardizer(
    X_train: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute columnwise mean/std using training data only.
    """
    X = X_train.to_numpy(dtype=np.float64)

    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    zero_variance = std == 0.0
    if np.any(zero_variance):
        std = std.copy()
        std[zero_variance] = 1.0

    return mean, std


def standardize(
    X: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = X.to_numpy(dtype=np.float64)
    return (values - mean) / std


def append_metrics(
    rows: list[dict],
    horizon: int,
    split_name: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_rmse: float,
) -> None:
    metrics = compute_metrics(y_true, y_pred)

    if model_name == "Persistence":
        skill = 0.0
    else:
        skill = 1.0 - metrics["rmse"] / persistence_rmse

    rows.append(
        {
            "horizon_hours": horizon,
            "split": split_name,
            "model": model_name,
            "n_samples": len(y_true),
            "mae_C": metrics["mae"],
            "rmse_C": metrics["rmse"],
            "bias_C": metrics["bias"],
            "skill_vs_persistence": skill,
        }
    )


def evaluate_split(
    split_df: pd.DataFrame,
    horizon: int,
    split_name: str,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    linear_model: MultipleLinearRegression,
    persistence_model: PersistenceModel,
    metrics_rows: list[dict],
    prediction_frames: list[pd.DataFrame],
) -> None:
    target_col = TARGET_COLS[horizon]

    X_raw = split_df[FEATURE_COLS]
    X_scaled = standardize(X_raw, scaler_mean, scaler_std)
    y_true = split_df[target_col].to_numpy(dtype=np.float64)

    persistence_pred = persistence_model.predict(X_raw)
    linear_pred = linear_model.predict(X_scaled)

    persistence_metrics = compute_metrics(y_true, persistence_pred)

    append_metrics(
        rows=metrics_rows,
        horizon=horizon,
        split_name=split_name,
        model_name="Persistence",
        y_true=y_true,
        y_pred=persistence_pred,
        persistence_rmse=persistence_metrics["rmse"],
    )

    append_metrics(
        rows=metrics_rows,
        horizon=horizon,
        split_name=split_name,
        model_name="MultipleLinearRegression",
        y_true=y_true,
        y_pred=linear_pred,
        persistence_rmse=persistence_metrics["rmse"],
    )

    predictions = pd.DataFrame(
        {
            "timestamp": split_df["timestamp"].to_numpy(),
            "valid_time": (
                split_df["timestamp"]
                + pd.to_timedelta(horizon, unit="h")
            ).to_numpy(),
            "split": split_name,
            "horizon_hours": horizon,
            "observed_temp_C": y_true,
            "persistence_pred_C": persistence_pred,
            "linear_pred_C": linear_pred,
        }
    )

    prediction_frames.append(predictions)


def train_baselines(
    df: pd.DataFrame,
    results_dir: Path,
    evaluate_test: bool = False,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    scaler_rows: list[dict] = []
    coefficient_rows: list[dict] = []

    print("\nBaseline temperature forecasting")
    print("--------------------------------")
    print(f"Rows in model-ready dataset: {len(df):,}")
    print(f"Features:                    {len(FEATURE_COLS)}")
    print(f"Horizons:                    {HORIZONS}")
    print()
    print("Split boundaries:")
    print(f"  Train:      target time < {VAL_START}")
    print(f"  Validation: {VAL_START} <= origin, target time < {TEST_START}")
    print(f"  Test:       {TEST_START} <= origin, target time < {TEST_END}")
    print()

    for horizon in HORIZONS:
        print(f"{horizon:>2} h horizon")

        splits = horizon_safe_split(df, horizon)

        train_df = splits["train"]
        val_df = splits["validation"]

        X_train_raw = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COLS[horizon]].to_numpy(dtype=np.float64)

        scaler_mean, scaler_std = fit_standardizer(X_train_raw)
        X_train_scaled = standardize(
            X_train_raw,
            scaler_mean,
            scaler_std,
        )

        linear_model = MultipleLinearRegression(fit_intercept=True)
        linear_model.fit(X_train_scaled, y_train)

        persistence_model = PersistenceModel()

        for feature, mean_value, std_value in zip(
            FEATURE_COLS,
            scaler_mean,
            scaler_std,
        ):
            scaler_rows.append(
                {
                    "horizon_hours": horizon,
                    "feature": feature,
                    "train_mean": mean_value,
                    "train_std": std_value,
                }
            )

        coefficient_rows.append(
            {
                "horizon_hours": horizon,
                "term": "intercept",
                "coefficient": linear_model.intercept_,
            }
        )

        for feature, coefficient in zip(
            FEATURE_COLS,
            linear_model.coef_,
        ):
            coefficient_rows.append(
                {
                    "horizon_hours": horizon,
                    "term": feature,
                    "coefficient": coefficient,
                }
            )

        evaluate_split(
            split_df=val_df,
            horizon=horizon,
            split_name="validation",
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            linear_model=linear_model,
            persistence_model=persistence_model,
            metrics_rows=metrics_rows,
            prediction_frames=prediction_frames,
        )

        if evaluate_test:
            evaluate_split(
                split_df=splits["test"],
                horizon=horizon,
                split_name="test",
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                linear_model=linear_model,
                persistence_model=persistence_model,
                metrics_rows=metrics_rows,
                prediction_frames=prediction_frames,
            )

        print(
            f"   train={len(train_df):,}, "
            f"validation={len(val_df):,}, "
            f"test={len(splits['test']):,}"
        )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    scaler_df = pd.DataFrame(scaler_rows)
    coefficients_df = pd.DataFrame(coefficient_rows)

    metrics_path = results_dir / "baseline_metrics.csv"
    predictions_path = results_dir / "baseline_predictions.csv"
    scaler_path = results_dir / "scaler_stats.csv"
    coefficients_path = results_dir / "ols_coefficients.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    scaler_df.to_csv(scaler_path, index=False)
    coefficients_df.to_csv(coefficients_path, index=False)

    print("\nValidation results")
    print("------------------")

    validation_metrics = metrics_df[
        metrics_df["split"] == "validation"
    ]

    display_cols = [
        "horizon_hours",
        "model",
        "mae_C",
        "rmse_C",
        "bias_C",
        "skill_vs_persistence",
    ]

    print(
        validation_metrics[display_cols]
        .round(4)
        .to_string(index=False)
    )

    if evaluate_test:
        print("\nTest results")
        print("------------")
        test_metrics = metrics_df[
            metrics_df["split"] == "test"
        ]
        print(
            test_metrics[display_cols]
            .round(4)
            .to_string(index=False)
        )
    else:
        print(
            "\nThe 2025 test set was not evaluated. "
            "This is intentional: keep it untouched while developing/tuning the MLP."
        )

    print("\nSaved:")
    print(f"  {metrics_path}")
    print(f"  {predictions_path}")
    print(f"  {scaler_path}")
    print(f"  {coefficients_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train persistence and multiple-linear-regression baselines "
            "for 1, 6, 12, 24, and 48 hour temperature forecasting."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to model-ready CSV (default: {DEFAULT_DATA_PATH})",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory for baseline outputs (default: {DEFAULT_RESULTS_DIR})",
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Also evaluate the held-out 2025 test set. "
            "Do not use this while tuning the later MLP."
        ),
    )

    args = parser.parse_args()

    df = load_model_ready(args.data)

    train_baselines(
        df=df,
        results_dir=args.results_dir,
        evaluate_test=args.evaluate_test,
    )


if __name__ == "__main__":
    main()
