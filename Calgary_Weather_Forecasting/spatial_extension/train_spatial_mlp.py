"""
train_spatial_mlp.py

Controlled neural-network comparison for the spatial extension.

Purpose
-------
Measure the incremental value of spatial station information while holding
the MLP training recipe fixed.

For each forecast horizon, train and compare:

    Local MLP:
        original 34 YYC predictors

    Spatial MLP:
        same 34 YYC predictors + 33 auxiliary-station predictors

The architecture and optimizer settings are copied from the configuration
selected in the original single-station project:

    hidden layer: 32 units
    activation: ReLU
    optimizer: Adam
    learning rate: 1e-3
    batch size: 64
    weight decay: 1e-5
    max epochs: 200
    early-stopping patience: 15
    seed: 2403

No new hyperparameter search is performed.

Default behaviour:
    train on pre-2024 data,
    early-stop using 2024 validation data,
    save the best checkpoints,
    report validation performance.

Protected test behaviour:
    python train_spatial_mlp.py --evaluate-test

This loads the already selected validation checkpoints and evaluates them
on 2025. It does not retrain or retune.

Inputs
------
data/processed/model_ready_spatial.csv
data/processed/spatial_feature_manifest.csv

Outputs
-------
results/spatial_mlp/validation_metrics.csv
results/spatial_mlp/spatial_gain_over_local.csv
results/spatial_mlp/training_history.csv
results/spatial_mlp/preprocessing_stats.csv
results/spatial_mlp/checkpoints/local_mlp_<h>h.pt
results/spatial_mlp/checkpoints/spatial_mlp_<h>h.pt

With --evaluate-test:
results/spatial_mlp/final_test_metrics.csv
results/spatial_mlp/final_test_spatial_gain.csv
results/spatial_mlp/final_test_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data/processed/model_ready_spatial.csv"
MANIFEST_PATH = BASE_DIR / "data/processed/spatial_feature_manifest.csv"

RESULTS_DIR = BASE_DIR / "results/spatial_mlp"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"

HORIZONS = (1, 6, 12, 24, 48)

TRAIN_END = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_START = pd.Timestamp("2024-01-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-01-01 00:00:00")
TEST_START = pd.Timestamp("2025-01-01 00:00:00")
TEST_END = pd.Timestamp("2026-01-01 00:00:00")

SEED = 2403
HIDDEN_DIM = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 200
PATIENCE = 15


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TemperatureMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1),
        )

        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Metrics
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_data() -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run preprocess_spatial.py first."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run preprocess_spatial.py first."
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
            f"Expected 34 local features, found {len(local_features)}."
        )

    if len(spatial_features) != 67:
        raise ValueError(
            f"Expected 67 spatial-model features, "
            f"found {len(spatial_features)}."
        )

    return df, local_features, spatial_features


def split_masks(
    df: pd.DataFrame,
    horizon: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
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

    return train_mask, validation_mask, test_mask


# ---------------------------------------------------------------------------
# Leakage-safe preprocessing
# ---------------------------------------------------------------------------

def fit_feature_preprocessor(
    X_train: pd.DataFrame,
) -> dict[str, pd.Series]:
    medians = X_train.median(axis=0)

    if medians.isna().any():
        bad = medians.index[medians.isna()].tolist()
        raise ValueError(
            "Features entirely missing in training: "
            + ", ".join(bad)
        )

    X_imputed = X_train.fillna(medians)
    means = X_imputed.mean(axis=0)
    stds = X_imputed.std(axis=0, ddof=0)
    stds = stds.mask(stds == 0.0, 1.0)

    return {
        "medians": medians,
        "means": means,
        "stds": stds,
    }


def transform_features(
    X: pd.DataFrame,
    prep: dict[str, pd.Series],
) -> np.ndarray:
    Z = X.fillna(prep["medians"])
    Z = (Z - prep["means"]) / prep["stds"]

    arr = Z.to_numpy(dtype=np.float32)

    if not np.isfinite(arr).all():
        raise ValueError(
            "Non-finite predictor remained after preprocessing."
        )

    return arr


def fit_target_preprocessor(
    y_train: np.ndarray,
) -> tuple[float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))

    if std == 0.0:
        std = 1.0

    return mean, std


def transform_target(
    y: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    return ((y - mean) / std).astype(np.float32)


def inverse_target(
    y_scaled: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    return y_scaled * std + mean


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator if shuffle else None,
    )


@torch.no_grad()
def mse_on_array(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()

    X_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)

    pred = model(X_tensor)

    return float(
        torch.mean((pred - y_tensor) ** 2).item()
    )


@torch.no_grad()
def predict_scaled(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()

    X_tensor = torch.from_numpy(X).to(device)
    pred = model(X_tensor)

    return pred.cpu().numpy()


def train_one_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    checkpoint_path: Path,
    input_dim: int,
    device: torch.device,
) -> tuple[nn.Module, int, list[dict]]:
    set_seed(SEED)

    model = TemperatureMLP(
        input_dim=input_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MSELoss()

    loader = make_loader(
        X_train,
        y_train,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history: list[dict] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()

        batch_losses = []

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()

            batch_losses.append(
                float(loss.item())
            )

        train_loss = float(
            np.mean(batch_losses)
        )

        val_loss = mse_on_array(
            model=model,
            X=X_val,
            y=y_val,
            device=device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )

        if val_loss < best_val_loss - 1e-10:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "weight_decay": WEIGHT_DECAY,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "seed": SEED,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, best_epoch, history


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run train_spatial_mlp.py without --evaluate-test first."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = TemperatureMLP(
        input_dim=int(checkpoint["input_dim"])
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, checkpoint


# ---------------------------------------------------------------------------
# Model/horizon preparation
# ---------------------------------------------------------------------------

def prepare_horizon(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> dict:
    target_col = f"target_temp_{horizon}h"

    train_mask, val_mask, test_mask = split_masks(
        df,
        horizon,
    )

    train = df.loc[train_mask].copy()
    val = df.loc[val_mask].copy()
    test = df.loc[test_mask].copy()

    y_train = train[target_col].to_numpy(dtype=float)
    y_val = val[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    prep = fit_feature_preprocessor(
        train[feature_cols]
    )

    X_train = transform_features(
        train[feature_cols],
        prep,
    )
    X_val = transform_features(
        val[feature_cols],
        prep,
    )
    X_test = transform_features(
        test[feature_cols],
        prep,
    )

    y_mean, y_std = fit_target_preprocessor(
        y_train
    )

    return {
        "train_df": train,
        "val_df": val,
        "test_df": test,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_raw": y_train,
        "y_val_raw": y_val,
        "y_test_raw": y_test,
        "y_train_scaled": transform_target(
            y_train,
            y_mean,
            y_std,
        ),
        "y_val_scaled": transform_target(
            y_val,
            y_mean,
            y_std,
        ),
        "feature_prep": prep,
        "target_mean": y_mean,
        "target_std": y_std,
    }


def metric_row(
    horizon: int,
    split: str,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    persistence_pred: np.ndarray,
    best_epoch: int | None,
) -> dict:
    persistence_rmse = rmse(
        y_true,
        persistence_pred,
    )

    model_rmse = rmse(
        y_true,
        y_pred,
    )

    return {
        "horizon_hours": horizon,
        "split": split,
        "model": model_name,
        "n_samples": len(y_true),
        "mae_C": mae(y_true, y_pred),
        "rmse_C": model_rmse,
        "bias_C": bias(y_true, y_pred),
        "skill_vs_persistence": (
            1.0 - model_rmse / persistence_rmse
        ),
        "best_epoch": best_epoch,
    }


def gain_row(
    horizon: int,
    split: str,
    local_rmse: float,
    spatial_rmse: float,
    n_samples: int,
) -> dict:
    return {
        "horizon_hours": horizon,
        "split": split,
        "n_samples": n_samples,
        "local_mlp_rmse_C": local_rmse,
        "spatial_mlp_rmse_C": spatial_rmse,
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


# ---------------------------------------------------------------------------
# Validation training
# ---------------------------------------------------------------------------

def train_validation(
    df: pd.DataFrame,
    local_features: list[str],
    spatial_features: list[str],
    device: torch.device,
) -> None:
    metric_rows: list[dict] = []
    gain_rows: list[dict] = []
    history_rows: list[dict] = []
    preprocessing_rows: list[dict] = []

    for horizon in HORIZONS:
        print(f"\n{horizon} h horizon")
        print("-" * 40)

        model_predictions = {}
        best_epochs = {}

        for model_key, feature_cols in (
            ("local", local_features),
            ("spatial", spatial_features),
        ):
            prepared = prepare_horizon(
                df=df,
                feature_cols=feature_cols,
                horizon=horizon,
            )

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"{model_key}_mlp_{horizon}h.pt"
            )

            print(
                f"Training {model_key:7s} MLP "
                f"({len(feature_cols)} inputs) ...",
                end=" ",
            )

            model, best_epoch, history = train_one_model(
                X_train=prepared["X_train"],
                y_train=prepared["y_train_scaled"],
                X_val=prepared["X_val"],
                y_val=prepared["y_val_scaled"],
                checkpoint_path=checkpoint_path,
                input_dim=len(feature_cols),
                device=device,
            )

            pred_scaled = predict_scaled(
                model=model,
                X=prepared["X_val"],
                device=device,
            )

            pred = inverse_target(
                pred_scaled,
                prepared["target_mean"],
                prepared["target_std"],
            )

            y_true = prepared["y_val_raw"]
            persistence_pred = (
                prepared["val_df"]["temp_lag_0"]
                .to_numpy(dtype=float)
            )

            row = metric_row(
                horizon=horizon,
                split="validation",
                model_name=(
                    "Local MLP"
                    if model_key == "local"
                    else "Spatial MLP"
                ),
                y_true=y_true,
                y_pred=pred,
                persistence_pred=persistence_pred,
                best_epoch=best_epoch,
            )

            metric_rows.append(row)
            model_predictions[model_key] = pred
            best_epochs[model_key] = best_epoch

            for h in history:
                history_rows.append(
                    {
                        "horizon_hours": horizon,
                        "model": (
                            "Local MLP"
                            if model_key == "local"
                            else "Spatial MLP"
                        ),
                        **h,
                    }
                )

            prep = prepared["feature_prep"]
            train_df = prepared["train_df"]
            val_df = prepared["val_df"]

            for feature in feature_cols:
                preprocessing_rows.append(
                    {
                        "horizon_hours": horizon,
                        "model": (
                            "Local MLP"
                            if model_key == "local"
                            else "Spatial MLP"
                        ),
                        "feature": feature,
                        "train_median": float(
                            prep["medians"][feature]
                        ),
                        "train_mean_after_imputation": float(
                            prep["means"][feature]
                        ),
                        "train_std_after_imputation": float(
                            prep["stds"][feature]
                        ),
                        "train_missing_n": int(
                            train_df[feature].isna().sum()
                        ),
                        "validation_missing_n": int(
                            val_df[feature].isna().sum()
                        ),
                    }
                )

            print(
                f"best epoch={best_epoch}, "
                f"RMSE={row['rmse_C']:.3f} C"
            )

        # Same validation truth for local/spatial because the split is identical.
        prepared_local = prepare_horizon(
            df=df,
            feature_cols=local_features,
            horizon=horizon,
        )

        y_true = prepared_local["y_val_raw"]

        local_rmse = rmse(
            y_true,
            model_predictions["local"],
        )
        spatial_rmse = rmse(
            y_true,
            model_predictions["spatial"],
        )

        gain = gain_row(
            horizon=horizon,
            split="validation",
            local_rmse=local_rmse,
            spatial_rmse=spatial_rmse,
            n_samples=len(y_true),
        )

        gain_rows.append(gain)

        print(
            f"Spatial gain over local MLP: "
            f"{gain['rmse_reduction_percent']:.2f}%"
        )

    pd.DataFrame(metric_rows).to_csv(
        RESULTS_DIR / "validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(gain_rows).to_csv(
        RESULTS_DIR / "spatial_gain_over_local.csv",
        index=False,
    )

    pd.DataFrame(history_rows).to_csv(
        RESULTS_DIR / "training_history.csv",
        index=False,
    )

    pd.DataFrame(preprocessing_rows).to_csv(
        RESULTS_DIR / "preprocessing_stats.csv",
        index=False,
    )

    config = {
        "hidden_dim": HIDDEN_DIM,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "seed": SEED,
        "local_input_dim": len(local_features),
        "spatial_input_dim": len(spatial_features),
    }

    with open(
        RESULTS_DIR / "fixed_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    gains = pd.DataFrame(gain_rows)

    print("\nValidation spatial MLP gains")
    print("----------------------------")
    print(
        gains[
            [
                "horizon_hours",
                "local_mlp_rmse_C",
                "spatial_mlp_rmse_C",
                "rmse_reduction_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


# ---------------------------------------------------------------------------
# Protected final test
# ---------------------------------------------------------------------------

def evaluate_test(
    df: pd.DataFrame,
    local_features: list[str],
    spatial_features: list[str],
    device: torch.device,
) -> None:
    metric_rows: list[dict] = []
    gain_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for horizon in HORIZONS:
        print(f"Evaluating {horizon} h test horizon ...")

        predictions = {}
        test_reference = None

        for model_key, feature_cols in (
            ("local", local_features),
            ("spatial", spatial_features),
        ):
            prepared = prepare_horizon(
                df=df,
                feature_cols=feature_cols,
                horizon=horizon,
            )

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"{model_key}_mlp_{horizon}h.pt"
            )

            model, checkpoint = load_model(
                checkpoint_path=checkpoint_path,
                device=device,
            )

            pred_scaled = predict_scaled(
                model=model,
                X=prepared["X_test"],
                device=device,
            )

            pred = inverse_target(
                pred_scaled,
                prepared["target_mean"],
                prepared["target_std"],
            )

            predictions[model_key] = pred

            if test_reference is None:
                test_reference = prepared

            y_true = prepared["y_test_raw"]
            persistence_pred = (
                prepared["test_df"]["temp_lag_0"]
                .to_numpy(dtype=float)
            )

            metric_rows.append(
                metric_row(
                    horizon=horizon,
                    split="test",
                    model_name=(
                        "Local MLP"
                        if model_key == "local"
                        else "Spatial MLP"
                    ),
                    y_true=y_true,
                    y_pred=pred,
                    persistence_pred=persistence_pred,
                    best_epoch=int(
                        checkpoint["best_epoch"]
                    ),
                )
            )

        y_true = test_reference["y_test_raw"]
        persistence_pred = (
            test_reference["test_df"]["temp_lag_0"]
            .to_numpy(dtype=float)
        )

        local_rmse = rmse(
            y_true,
            predictions["local"],
        )
        spatial_rmse = rmse(
            y_true,
            predictions["spatial"],
        )

        gain_rows.append(
            gain_row(
                horizon=horizon,
                split="test",
                local_rmse=local_rmse,
                spatial_rmse=spatial_rmse,
                n_samples=len(y_true),
            )
        )

        for timestamp, y, p0, pl, ps in zip(
            test_reference["test_df"]["timestamp"],
            y_true,
            persistence_pred,
            predictions["local"],
            predictions["spatial"],
        ):
            prediction_rows.append(
                {
                    "timestamp": timestamp,
                    "horizon_hours": horizon,
                    "split": "test",
                    "observed_temp_C": y,
                    "persistence_temp_C": p0,
                    "local_mlp_temp_C": pl,
                    "spatial_mlp_temp_C": ps,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    gains_df = pd.DataFrame(gain_rows)
    predictions_df = pd.DataFrame(prediction_rows)

    metrics_df.to_csv(
        RESULTS_DIR / "final_test_metrics.csv",
        index=False,
    )

    gains_df.to_csv(
        RESULTS_DIR / "final_test_spatial_gain.csv",
        index=False,
    )

    predictions_df.to_csv(
        RESULTS_DIR / "final_test_predictions.csv",
        index=False,
    )

    print("\nFinal test spatial MLP gains")
    print("----------------------------")
    print(
        gains_df[
            [
                "horizon_hours",
                "local_mlp_rmse_C",
                "spatial_mlp_rmse_C",
                "rmse_reduction_percent",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def export_numpy_weights() -> None:
    """Export the same fitted affine layers for the NumPy-only portfolio check."""
    weights = {}
    for kind in ("local", "spatial"):
        for horizon in HORIZONS:
            path = CHECKPOINT_DIR / f"{kind}_mlp_{horizon}h.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            for name, tensor in checkpoint["model_state_dict"].items():
                weights[f"{kind}_{horizon}_{name}"] = tensor.detach().numpy()
    np.savez_compressed(CHECKPOINT_DIR / "matched_weights.npz", **weights)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed local and spatial MLPs "
            "without additional hyperparameter tuning."
        )
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-numpy", action="store_true",
                      help="Export saved PyTorch weights for the NumPy verification script.")
    mode.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Load validation-selected checkpoints and "
            "evaluate the protected 2025 test set."
        ),
    )

    args = parser.parse_args()
    if args.export_numpy:
        export_numpy_weights()
        return

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df, local_features, spatial_features = load_data()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(
        f"Local inputs: {len(local_features)} | "
        f"Spatial inputs: {len(spatial_features)}"
    )

    if args.evaluate_test:
        evaluate_test(
            df=df,
            local_features=local_features,
            spatial_features=spatial_features,
            device=device,
        )
    else:
        train_validation(
            df=df,
            local_features=local_features,
            spatial_features=spatial_features,
            device=device,
        )
        export_numpy_weights()


if __name__ == "__main__":
    main()
