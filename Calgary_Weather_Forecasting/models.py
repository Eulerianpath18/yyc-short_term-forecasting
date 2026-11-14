"""
models.py

Forecasting models for the Calgary temperature forecasting project.

Models
------
1. PersistenceModel
2. MultipleLinearRegression
3. TemperatureMLP
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn


HORIZONS = (1, 6, 12, 24, 48)

FEATURE_COLS = [
    *[f"temp_lag_{i}" for i in range(24)],
    "dew_point",
    "relative_humidity",
    "station_pressure",
    "wind_speed",
    "wind_dir_sin",
    "wind_dir_cos",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]

TARGET_COLS = {
    h: f"target_temp_{h}h"
    for h in HORIZONS
}


class PersistenceModel:
    """
    Persistence baseline:
        T_hat(t + h) = T(t)
    """

    def __init__(self, current_temperature_col: str = "temp_lag_0"):
        self.current_temperature_col = current_temperature_col

    def fit(self, X=None, y=None) -> "PersistenceModel":
        return self

    def predict(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            if self.current_temperature_col not in X.columns:
                raise ValueError(
                    f"DataFrame must contain '{self.current_temperature_col}'."
                )
            return X[self.current_temperature_col].to_numpy(dtype=np.float64)

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        return X[:, 0].copy()


class MultipleLinearRegression:
    """
    Ordinary multiple linear regression fitted by least squares.
    """

    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
        self.intercept_: float | None = None
        self.coef_: np.ndarray | None = None
        self.rank_: int | None = None
        self.singular_values_: np.ndarray | None = None
        self.is_fitted_: bool = False

    @staticmethod
    def _to_numpy(X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        else:
            X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        return X

    def fit(self, X, y) -> "MultipleLinearRegression":
        X = self._to_numpy(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if not np.all(np.isfinite(y)):
            raise ValueError("y contains NaN or infinite values.")

        if self.fit_intercept:
            design_matrix = np.column_stack([np.ones(X.shape[0]), X])
        else:
            design_matrix = X

        beta, _, rank, singular_values = np.linalg.lstsq(
            design_matrix, y, rcond=None
        )

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.rank_ = int(rank)
        self.singular_values_ = singular_values
        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before prediction.")

        X = self._to_numpy(X)

        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                f"Expected {self.coef_.shape[0]} features, "
                f"but received {X.shape[1]}."
            )

        return self.intercept_ + X @ self.coef_


class TemperatureMLP(nn.Module):
    """
    Feed-forward neural network for one temperature forecast horizon.

    Example:
        TemperatureMLP(input_dim=34, hidden_dims=(64, 32))

    No batch normalization or dropout is used in the initial experiment.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for width in hidden_dims:
            linear = nn.Linear(previous_dim, width)
            nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            nn.init.zeros_(linear.bias)

            layers.extend([linear, nn.ReLU()])
            previous_dim = width

        output = nn.Linear(previous_dim, 1)
        nn.init.kaiming_normal_(output.weight, nonlinearity="linear")
        nn.init.zeros_(output.bias)
        layers.append(output)

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError("Missing feature columns: " + ", ".join(missing))

    return df[feature_cols].copy()


def get_target(
    df: pd.DataFrame,
    horizon: int,
) -> np.ndarray:
    if horizon not in TARGET_COLS:
        raise ValueError(f"horizon must be one of {HORIZONS}; got {horizon}.")

    target_col = TARGET_COLS[horizon]
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}'.")

    return df[target_col].to_numpy(dtype=np.float64)
