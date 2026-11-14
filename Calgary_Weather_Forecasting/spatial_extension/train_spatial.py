"""
train_spatial.py

Baseline experiment for the multi-station extension.

The purpose of this script is to isolate the value of spatial information.
For each forecast horizon it evaluates, on exactly the same timestamps:

    1. Persistence
    2. Local OLS       -- original 34 YYC predictors
    3. Spatial OLS     -- the same 34 predictors + 33 auxiliary predictors

Auxiliary missing values are imputed using medians estimated from the TRAINING
period only. All predictors are then standardized using TRAINING-period means
and standard deviations only.

By default, the script evaluates on the 2024 validation period and leaves the
2025 test period untouched.

Usage
-----
From spatial_extension/:

    python train_spatial.py

Final test evaluation, only after model decisions are frozen:

    python train_spatial.py --evaluate-test

Inputs
------
data/processed/model_ready_spatial.csv
data/processed/spatial_feature_manifest.csv

Outputs
-------
results/spatial_baselines/baseline_metrics.csv
results/spatial_baselines/spatial_gain_over_local.csv
results/spatial_baselines/baseline_predictions.csv
results/spatial_baselines/preprocessing_stats.csv
results/spatial_baselines/local_ols_coefficients.csv
results/spatial_baselines/spatial_ols_coefficients.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/processed/model_ready_spatial.csv"
MANIFEST_PATH = BASE_DIR / "data/processed/spatial_feature_manifest.csv"

RESULTS_DIR = BASE_DIR / "results/spatial_baselines"

HORIZONS = (1, 6, 12, 24, 48)

TRAIN_END = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_START = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.sqrt(np.mean((y_pred - y_true) ** 2))
    )


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(np.abs(y_pred - y_true))
    )


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(
        np.mean(y_pred - y_true)
    )


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_rmse: float,
) -> dict[str, float]:
    model_rmse = rmse(y_true, y_pred)

    if persistence_rmse > 0:
        skill = 1.0 - model_rmse / persistence_rmse
    else:
        skill = np.nan

    return {
        "mae_C": mae(y_true, y_pred),
        "rmse_C": model_rmse,
        "bias_C": bias(y_true, y_pred),
        "skill_vs_persistence": skill,
    }


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

class OLS:
    """Ordinary least squares with an explicit intercept."""

    def __init__(self) -> None:
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "OLS":
        X_design = np.column_stack(
            [np.ones(len(X), dtype=float), X]
        )

        beta, *_ = np.linalg.lstsq(
            X_design,
            y,
            rcond=None,
        )

        self.intercept_ = float(beta[0])
        self.coef_ = np.asarray(beta[1:], dtype=float)
        return self

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        if self.intercept_ is None or self.coef_ is None:
            raise RuntimeError("Model has not been fitted.")

        return self.intercept_ + X @ self.coef_


# ---------------------------------------------------------------------------
# Data loading and split construction
# ---------------------------------------------------------------------------

def load_data() -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. "
            "Run preprocess_spatial.py first."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. "
            "Run preprocess_spatial.py first."
        )

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
    )

    manifest = pd.read_csv(MANIFEST_PATH)

    local_features = (
        manifest.loc[
            manifest["feature_group"] == "local",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    spatial_only = (
        manifest.loc[
            manifest["feature_group"] == "spatial",
            "feature",
        ]
        .astype(str)
        .tolist()
    )

    spatial_features = local_features + spatial_only

    if len(local_features) != 34:
        raise ValueError(
            f"Expected 34 local features; found "
            f"{len(local_features)}."
        )

    if len(spatial_only) != 33:
        raise ValueError(
            f"Expected 33 auxiliary features; found "
            f"{len(spatial_only)}."
        )

    if len(spatial_features) != 67:
        raise ValueError(
            f"Expected 67 total predictors; found "
            f"{len(spatial_features)}."
        )

    expected = set(spatial_features)
    missing_cols = expected - set(df.columns)

    if missing_cols:
        raise ValueError(
            "Spatial dataset is missing predictor columns: "
            + ", ".join(sorted(missing_cols))
        )

    return df, local_features, spatial_features


def get_masks(
    df: pd.DataFrame,
    horizon: int,
    evaluate_test: bool,
) -> tuple[
    pd.Series,
    pd.Series,
    str,
]:
    """
    Horizon-safe chronological split.

    The target time must remain inside the split to which the forecast origin
    is assigned.
    """
    origin = df["timestamp"]
    target_time = origin + pd.to_timedelta(
        horizon,
        unit="h",
    )

    train_mask = target_time < TRAIN_END

    validation_mask = (
        (origin >= VALIDATION_START)
        & (target_time < VALIDATION_END)
    )

    test_mask = (
        (origin >= TEST_START)
        & (target_time < TEST_END)
    )

    if evaluate_test:
        evaluation_mask = test_mask
        split_name = "test"
    else:
        evaluation_mask = validation_mask
        split_name = "validation"

    return train_mask, evaluation_mask, split_name


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing
# ---------------------------------------------------------------------------

def fit_preprocessor(
    X_train_raw: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Fit median imputation + z-score scaling using training data only.

    Returns
    -------
    medians, means, stds
    """
    medians = X_train_raw.median(axis=0)

    if medians.isna().any():
        bad = medians.index[medians.isna()].tolist()
        raise ValueError(
            "At least one feature is entirely missing in the "
            "training period: "
            + ", ".join(bad)
        )

    X_train_imputed = X_train_raw.fillna(medians)

    means = X_train_imputed.mean(axis=0)
    stds = X_train_imputed.std(axis=0, ddof=0)

    # A constant feature carries no information. Setting its scale to 1 leaves
    # the standardized column equal to zero rather than dividing by zero.
    stds = stds.mask(stds == 0.0, 1.0)

    return medians, means, stds


def transform(
    X_raw: pd.DataFrame,
    medians: pd.Series,
    means: pd.Series,
    stds: pd.Series,
) -> np.ndarray:
    X = X_raw.fillna(medians)
    X = (X - means) / stds

    arr = X.to_numpy(dtype=float)

    if not np.isfinite(arr).all():
        raise ValueError(
            "Non-finite value remained after imputation/scaling."
        )

    return arr


# ---------------------------------------------------------------------------
# Main horizon experiment
# ---------------------------------------------------------------------------

def run_horizon(
    df: pd.DataFrame,
    local_features: list[str],
    spatial_features: list[str],
    horizon: int,
    evaluate_test: bool,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    list[dict],
    list[dict],
]:
    target_col = f"target_temp_{horizon}h"

    if target_col not in df.columns:
        raise ValueError(
            f"Missing target column {target_col}."
        )

    train_mask, eval_mask, split_name = get_masks(
        df,
        horizon=horizon,
        evaluate_test=evaluate_test,
    )

    train = df.loc[train_mask].copy()
    evaluation = df.loc[eval_mask].copy()

    if train.empty:
        raise ValueError(
            f"No training rows for {horizon} h."
        )

    if evaluation.empty:
        raise ValueError(
            f"No {split_name} rows for {horizon} h."
        )

    # All three models are evaluated on exactly these same rows.
    y_train = train[target_col].to_numpy(dtype=float)
    y_eval = evaluation[target_col].to_numpy(dtype=float)

    if not np.isfinite(y_train).all():
        raise ValueError(
            f"Non-finite training targets at {horizon} h."
        )

    if not np.isfinite(y_eval).all():
        raise ValueError(
            f"Non-finite evaluation targets at {horizon} h."
        )

    # Fit ONE preprocessing transformation for the full 67-dimensional feature
    # space. The local model simply uses the corresponding local subset.
    medians, means, stds = fit_preprocessor(
        train[spatial_features]
    )

    X_train_spatial = transform(
        train[spatial_features],
        medians=medians,
        means=means,
        stds=stds,
    )

    X_eval_spatial = transform(
        evaluation[spatial_features],
        medians=medians,
        means=means,
        stds=stds,
    )

    local_indices = [
        spatial_features.index(col)
        for col in local_features
    ]

    X_train_local = X_train_spatial[:, local_indices]
    X_eval_local = X_eval_spatial[:, local_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    persistence_pred = (
        evaluation["temp_lag_0"]
        .to_numpy(dtype=float)
    )

    persistence_rmse = rmse(
        y_eval,
        persistence_pred,
    )

    # ------------------------------------------------------------------
    # Local OLS
    # ------------------------------------------------------------------

    local_model = OLS().fit(
        X_train_local,
        y_train,
    )
    local_pred = local_model.predict(
        X_eval_local
    )

    # ------------------------------------------------------------------
    # Spatial OLS
    # ------------------------------------------------------------------

    spatial_model = OLS().fit(
        X_train_spatial,
        y_train,
    )
    spatial_pred = spatial_model.predict(
        X_eval_spatial
    )

    metric_rows = []

    for name, pred in (
        ("Persistence", persistence_pred),
        ("Local OLS", local_pred),
        ("Spatial OLS", spatial_pred),
    ):
        row = {
            "horizon_hours": horizon,
            "split": split_name,
            "model": name,
            "n_samples": len(evaluation),
        }

        row.update(
            metrics(
                y_true=y_eval,
                y_pred=pred,
                persistence_rmse=persistence_rmse,
            )
        )

        metric_rows.append(row)

    local_rmse = rmse(y_eval, local_pred)
    spatial_rmse = rmse(y_eval, spatial_pred)

    gain_rows = [
        {
            "horizon_hours": horizon,
            "split": split_name,
            "n_samples": len(evaluation),
            "local_ols_rmse_C": local_rmse,
            "spatial_ols_rmse_C": spatial_rmse,
            "rmse_reduction_C": (
                local_rmse - spatial_rmse
            ),
            "rmse_reduction_percent": (
                100.0
                * (local_rmse - spatial_rmse)
                / local_rmse
            ),
            "spatial_skill_vs_local": (
                1.0 - spatial_rmse / local_rmse
            ),
        }
    ]

    prediction_rows = []

    for timestamp, y, p0, pl, ps in zip(
        evaluation["timestamp"],
        y_eval,
        persistence_pred,
        local_pred,
        spatial_pred,
    ):
        prediction_rows.append(
            {
                "timestamp": timestamp,
                "horizon_hours": horizon,
                "split": split_name,
                "observed_temp_C": y,
                "persistence_temp_C": p0,
                "local_ols_temp_C": pl,
                "spatial_ols_temp_C": ps,
            }
        )

    preprocessing_rows = []

    for feature in spatial_features:
        preprocessing_rows.append(
            {
                "horizon_hours": horizon,
                "feature": feature,
                "train_median": float(medians[feature]),
                "train_mean_after_imputation": float(means[feature]),
                "train_std_after_imputation": float(stds[feature]),
                "train_missing_n": int(
                    train[feature].isna().sum()
                ),
                "eval_missing_n": int(
                    evaluation[feature].isna().sum()
                ),
            }
        )

    local_coef_rows = [
        {
            "horizon_hours": horizon,
            "feature": "(intercept)",
            "coefficient": local_model.intercept_,
        }
    ]

    local_coef_rows.extend(
        {
            "horizon_hours": horizon,
            "feature": feature,
            "coefficient": float(coef),
        }
        for feature, coef in zip(
            local_features,
            local_model.coef_,
        )
    )

    spatial_coef_rows = [
        {
            "horizon_hours": horizon,
            "feature": "(intercept)",
            "coefficient": spatial_model.intercept_,
        }
    ]

    spatial_coef_rows.extend(
        {
            "horizon_hours": horizon,
            "feature": feature,
            "coefficient": float(coef),
        }
        for feature, coef in zip(
            spatial_features,
            spatial_model.coef_,
        )
    )

    return (
        metric_rows,
        gain_rows,
        prediction_rows,
        preprocessing_rows,
        local_coef_rows + spatial_coef_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare persistence, local OLS, and spatial OLS "
            "for Calgary temperature forecasting."
        )
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate the protected 2025 test period. "
            "Do not use this during model development."
        ),
    )

    args = parser.parse_args()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, local_features, spatial_features = load_data()

    all_metrics: list[dict] = []
    all_gains: list[dict] = []
    all_predictions: list[dict] = []
    all_preprocessing: list[dict] = []
    all_local_coefficients: list[dict] = []
    all_spatial_coefficients: list[dict] = []

    for horizon in HORIZONS:
        print(
            f"Fitting {horizon:>2} h baseline models ...",
            end=" ",
        )

        (
            metric_rows,
            gain_rows,
            prediction_rows,
            preprocessing_rows,
            combined_coefficients,
        ) = run_horizon(
            df=df,
            local_features=local_features,
            spatial_features=spatial_features,
            horizon=horizon,
            evaluate_test=args.evaluate_test,
        )

        all_metrics.extend(metric_rows)
        all_gains.extend(gain_rows)
        all_predictions.extend(prediction_rows)
        all_preprocessing.extend(preprocessing_rows)

        # Separate the coefficient rows using feature membership.
        n_local = 1 + len(local_features)
        all_local_coefficients.extend(
            combined_coefficients[:n_local]
        )
        all_spatial_coefficients.extend(
            combined_coefficients[n_local:]
        )

        gain = gain_rows[0]
        print(
            f"local RMSE={gain['local_ols_rmse_C']:.3f} C, "
            f"spatial RMSE={gain['spatial_ols_rmse_C']:.3f} C, "
            f"gain={gain['rmse_reduction_percent']:.2f}%"
        )

    metrics_df = pd.DataFrame(all_metrics)
    gains_df = pd.DataFrame(all_gains)
    predictions_df = pd.DataFrame(all_predictions)
    preprocessing_df = pd.DataFrame(all_preprocessing)
    local_coef_df = pd.DataFrame(all_local_coefficients)
    spatial_coef_df = pd.DataFrame(all_spatial_coefficients)

    prefix = "final_test" if args.evaluate_test else "validation"
    metrics_path = RESULTS_DIR / f"{prefix}_baseline_metrics.csv"
    gains_path = RESULTS_DIR / f"{prefix}_spatial_gain_over_local.csv"
    predictions_path = RESULTS_DIR / f"{prefix}_baseline_predictions.csv"
    preprocessing_path = RESULTS_DIR / f"{prefix}_preprocessing_stats.csv"
    local_coef_path = RESULTS_DIR / f"{prefix}_local_ols_coefficients.csv"
    spatial_coef_path = RESULTS_DIR / f"{prefix}_spatial_ols_coefficients.csv"

    metrics_df.to_csv(metrics_path, index=False)
    gains_df.to_csv(gains_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    preprocessing_df.to_csv(preprocessing_path, index=False)
    local_coef_df.to_csv(local_coef_path, index=False)
    spatial_coef_df.to_csv(spatial_coef_path, index=False)

    split_name = (
        "test" if args.evaluate_test
        else "validation"
    )

    print()
    print(f"{split_name.capitalize()} spatial baseline results")
    print("-" * 40)
    print(
        gains_df[
            [
                "horizon_hours",
                "local_ols_rmse_C",
                "spatial_ols_rmse_C",
                "rmse_reduction_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print()
    print("Saved:")
    print(f"  {metrics_path}")
    print(f"  {gains_path}")
    print(f"  {predictions_path}")
    print(f"  {preprocessing_path}")
    print(f"  {local_coef_path}")
    print(f"  {spatial_coef_path}")


if __name__ == "__main__":
    main()
