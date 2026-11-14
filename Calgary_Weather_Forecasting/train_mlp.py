"""
train_mlp.py

Train and tune PyTorch MLP temperature forecasts for horizons
{1, 6, 12, 24, 48} hours.

Experimental protocol
---------------------
Training:   pre-2024 data
Validation: 2024
Test:       2025 (protected unless --evaluate-test is supplied)

All feature and target normalization statistics are fitted using training
data only.

Hyperparameter selection is deliberately small and staged:

Stage 1: learning rate
    {1e-4, 3e-4, 1e-3}

Stage 2: architecture
    {(32,), (64, 32), (128, 64)}

Stage 3: batch size
    {64, 256}

Stage 4: Adam weight decay
    {0, 1e-5, 1e-4}

For a configuration c, one separate MLP is trained for each forecast horizon.
The common hyperparameter configuration is ranked by

    Q_c = mean_h[ RMSE_MLP,c(h) / RMSE_OLS(h) ]

on the 2024 validation set.

Lower Q is better. The 2025 test set plays no role in hyperparameter selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import (
    FEATURE_COLS,
    HORIZONS,
    TARGET_COLS,
    MultipleLinearRegression,
    PersistenceModel,
    TemperatureMLP,
)
from train import (
    TEST_END,
    TEST_START,
    VAL_START,
    compute_metrics,
    fit_standardizer,
    horizon_safe_split,
    load_model_ready,
    rmse,
    standardize,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data/processed/model_ready.csv"
DEFAULT_BASELINE_METRICS = BASE_DIR / "results/baselines/baseline_metrics.csv"
DEFAULT_RESULTS_DIR = BASE_DIR / "results/mlp"

DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 15
DEFAULT_MIN_DELTA = 1e-5
DEFAULT_SEED = 2403


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    batch_size: int = 256
    weight_decay: float = 0.0

    def key(self) -> tuple:
        return (
            self.hidden_dims,
            self.learning_rate,
            self.batch_size,
            self.weight_decay,
        )


@dataclass
class HorizonResult:
    horizon: int
    validation_rmse_C: float
    validation_mae_C: float
    validation_bias_C: float
    ols_validation_rmse_C: float
    persistence_validation_rmse_C: float
    best_epoch: int
    best_val_loss_standardized: float
    model_state: dict
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    history: list[dict[str, float]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def target_standardizer(y_train: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(y_train))
    std = float(np.std(y_train, ddof=0))
    if std == 0.0:
        std = 1.0
    return mean, std


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


@torch.no_grad()
def predict_standardized(
    model: TemperatureMLP,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()

    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    predictions = []

    for (xb,) in loader:
        xb = xb.to(device)
        predictions.append(model(xb).cpu().numpy())

    return np.concatenate(predictions)


@torch.no_grad()
def mean_mse(
    model: TemperatureMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        n = xb.shape[0]
        total_loss += float(loss.item()) * n
        total_n += n

    return total_loss / total_n


def train_one_horizon(
    df: pd.DataFrame,
    horizon: int,
    config: MLPConfig,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> HorizonResult:
    splits = horizon_safe_split(df, horizon)
    train_df = splits["train"]
    val_df = splits["validation"]

    X_train_raw = train_df[FEATURE_COLS]
    X_val_raw = val_df[FEATURE_COLS]

    y_train = train_df[TARGET_COLS[horizon]].to_numpy(dtype=np.float64)
    y_val = val_df[TARGET_COLS[horizon]].to_numpy(dtype=np.float64)

    # Fit all preprocessing statistics using training data only.
    x_mean, x_std = fit_standardizer(X_train_raw)
    X_train = standardize(X_train_raw, x_mean, x_std).astype(np.float32)
    X_val = standardize(X_val_raw, x_mean, x_std).astype(np.float32)

    y_mean, y_std = target_standardizer(y_train)
    y_train_std = ((y_train - y_mean) / y_std).astype(np.float32)
    y_val_std = ((y_val - y_mean) / y_std).astype(np.float32)

    # Baseline RMSE values for configuration scoring.
    linear_model = MultipleLinearRegression().fit(X_train, y_train)
    ols_val_pred = linear_model.predict(X_val)
    ols_val_rmse = rmse(y_val, ols_val_pred)

    persistence_model = PersistenceModel()
    persistence_val_pred = persistence_model.predict(X_val_raw)
    persistence_val_rmse = rmse(y_val, persistence_val_pred)

    # Fixed seed for fair comparisons between hyperparameter configurations.
    set_seed(seed + horizon)

    model = TemperatureMLP(
        input_dim=len(FEATURE_COLS),
        hidden_dims=config.hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.MSELoss()

    train_loader = make_loader(
        X_train,
        y_train_std,
        batch_size=config.batch_size,
        shuffle=True,
        seed=seed + horizon,
    )

    val_loader = make_loader(
        X_val,
        y_val_std,
        batch_size=max(config.batch_size, 1024),
        shuffle=False,
        seed=seed + horizon,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()

            n = xb.shape[0]
            train_loss_sum += float(loss.item()) * n
            train_n += n

        train_loss = train_loss_sum / train_n
        val_loss = mean_mse(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_mse_standardized": train_loss,
                "validation_mse_standardized": val_loss,
            }
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)

    val_pred_std = predict_standardized(
        model,
        X_val,
        device=device,
    )
    val_pred_C = y_mean + y_std * val_pred_std

    val_metrics = compute_metrics(y_val, val_pred_C)

    return HorizonResult(
        horizon=horizon,
        validation_rmse_C=val_metrics["rmse"],
        validation_mae_C=val_metrics["mae"],
        validation_bias_C=val_metrics["bias"],
        ols_validation_rmse_C=ols_val_rmse,
        persistence_validation_rmse_C=persistence_val_rmse,
        best_epoch=best_epoch,
        best_val_loss_standardized=best_val_loss,
        model_state=copy.deepcopy(model.state_dict()),
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        history=history,
    )


def config_score(results: dict[int, HorizonResult]) -> float:
    ratios = [
        results[h].validation_rmse_C
        / results[h].ols_validation_rmse_C
        for h in HORIZONS
    ]
    return float(np.mean(ratios))


def evaluate_config(
    df: pd.DataFrame,
    config: MLPConfig,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> tuple[float, dict[int, HorizonResult]]:
    print(
        "\nConfig:",
        f"hidden={config.hidden_dims},",
        f"lr={config.learning_rate:g},",
        f"batch={config.batch_size},",
        f"weight_decay={config.weight_decay:g}",
    )

    results: dict[int, HorizonResult] = {}

    for horizon in HORIZONS:
        result = train_one_horizon(
            df=df,
            horizon=horizon,
            config=config,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            seed=seed,
        )
        results[horizon] = result

        ratio = (
            result.validation_rmse_C
            / result.ols_validation_rmse_C
        )

        print(
            f"  {horizon:>2} h: "
            f"MLP RMSE={result.validation_rmse_C:.4f} C, "
            f"OLS RMSE={result.ols_validation_rmse_C:.4f} C, "
            f"ratio={ratio:.4f}, "
            f"best epoch={result.best_epoch}"
        )

    score = config_score(results)
    print(f"  Q = {score:.5f}")

    return score, results


def staged_search(
    df: pd.DataFrame,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
    seed: int,
) -> tuple[
    MLPConfig,
    dict[int, HorizonResult],
    pd.DataFrame,
]:
    cache: dict[
        tuple,
        tuple[float, dict[int, HorizonResult]],
    ] = {}
    rows: list[dict] = []

    def run(config: MLPConfig, stage: str):
        key = config.key()

        if key not in cache:
            score, results = evaluate_config(
                df=df,
                config=config,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                min_delta=min_delta,
                seed=seed,
            )
            cache[key] = (score, results)
        else:
            score, results = cache[key]

        rows.append(
            {
                "stage": stage,
                "hidden_dims": "-".join(map(str, config.hidden_dims)),
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "weight_decay": config.weight_decay,
                "Q_mean_rmse_ratio_to_OLS": score,
            }
        )

        return score, results

    # Stage 1: learning rate
    stage1 = [
        MLPConfig(
            hidden_dims=(64, 32),
            learning_rate=lr,
            batch_size=256,
            weight_decay=0.0,
        )
        for lr in (1e-4, 3e-4, 1e-3)
    ]

    scored = [(run(c, "learning_rate")[0], c) for c in stage1]
    best_lr_config = min(scored, key=lambda item: item[0])[1]

    # Stage 2: architecture
    stage2 = [
        MLPConfig(
            hidden_dims=hidden,
            learning_rate=best_lr_config.learning_rate,
            batch_size=256,
            weight_decay=0.0,
        )
        for hidden in ((32,), (64, 32), (128, 64))
    ]

    scored = [(run(c, "architecture")[0], c) for c in stage2]
    best_arch_config = min(scored, key=lambda item: item[0])[1]

    # Stage 3: batch size
    stage3 = [
        MLPConfig(
            hidden_dims=best_arch_config.hidden_dims,
            learning_rate=best_arch_config.learning_rate,
            batch_size=batch,
            weight_decay=0.0,
        )
        for batch in (64, 256)
    ]

    scored = [(run(c, "batch_size")[0], c) for c in stage3]
    best_batch_config = min(scored, key=lambda item: item[0])[1]

    # Stage 4: weight decay
    stage4 = [
        MLPConfig(
            hidden_dims=best_batch_config.hidden_dims,
            learning_rate=best_batch_config.learning_rate,
            batch_size=best_batch_config.batch_size,
            weight_decay=wd,
        )
        for wd in (0.0, 1e-5, 1e-4)
    ]

    scored = [(run(c, "weight_decay")[0], c) for c in stage4]
    best_config = min(scored, key=lambda item: item[0])[1]

    best_score, best_results = cache[best_config.key()]

    search_df = (
        pd.DataFrame(rows)
        .drop_duplicates(
            subset=[
                "hidden_dims",
                "learning_rate",
                "batch_size",
                "weight_decay",
            ],
            keep="first",
        )
        .sort_values("Q_mean_rmse_ratio_to_OLS")
        .reset_index(drop=True)
    )

    print("\nSelected configuration")
    print("----------------------")
    print(best_config)
    print(f"Q = {best_score:.5f}")

    return best_config, best_results, search_df


def save_selected_run(
    results_dir: Path,
    config: MLPConfig,
    results: dict[int, HorizonResult],
    search_df: pd.DataFrame,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    config_dict["hidden_dims"] = list(config.hidden_dims)

    with open(results_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    search_df.to_csv(
        results_dir / "hyperparameter_search.csv",
        index=False,
    )

    metric_rows = []
    scaler_rows = []
    history_frames = []

    for horizon, result in results.items():
        checkpoint = {
            "horizon": horizon,
            "config": config_dict,
            "input_dim": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
            "model_state_dict": result.model_state,
            "x_mean": result.x_mean,
            "x_std": result.x_std,
            "y_mean": result.y_mean,
            "y_std": result.y_std,
            "best_epoch": result.best_epoch,
        }

        torch.save(
            checkpoint,
            checkpoints_dir / f"mlp_{horizon}h.pt",
        )

        metric_rows.append(
            {
                "horizon_hours": horizon,
                "split": "validation",
                "model": "MLP",
                "mae_C": result.validation_mae_C,
                "rmse_C": result.validation_rmse_C,
                "bias_C": result.validation_bias_C,
                "ols_rmse_C": result.ols_validation_rmse_C,
                "persistence_rmse_C": result.persistence_validation_rmse_C,
                "rmse_ratio_to_OLS": (
                    result.validation_rmse_C
                    / result.ols_validation_rmse_C
                ),
                "skill_vs_persistence": (
                    1.0
                    - result.validation_rmse_C
                    / result.persistence_validation_rmse_C
                ),
                "best_epoch": result.best_epoch,
            }
        )

        for feature, mean_value, std_value in zip(
            FEATURE_COLS,
            result.x_mean,
            result.x_std,
        ):
            scaler_rows.append(
                {
                    "horizon_hours": horizon,
                    "variable": feature,
                    "mean": mean_value,
                    "std": std_value,
                }
            )

        scaler_rows.extend(
            [
                {
                    "horizon_hours": horizon,
                    "variable": "TARGET",
                    "mean": result.y_mean,
                    "std": result.y_std,
                }
            ]
        )

        history_df = pd.DataFrame(result.history)
        history_df.insert(0, "horizon_hours", horizon)
        history_frames.append(history_df)

    pd.DataFrame(metric_rows).to_csv(
        results_dir / "mlp_validation_metrics.csv",
        index=False,
    )

    pd.DataFrame(scaler_rows).to_csv(
        results_dir / "mlp_scaler_stats.csv",
        index=False,
    )

    pd.concat(history_frames, ignore_index=True).to_csv(
        results_dir / "training_history.csv",
        index=False,
    )


def load_selected_run(
    results_dir: Path,
    device: torch.device,
) -> tuple[MLPConfig, dict[int, dict]]:
    config_path = results_dir / "best_config.json"
    checkpoints_dir = results_dir / "checkpoints"

    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} does not exist. Run hyperparameter search first."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = json.load(f)

    config = MLPConfig(
        hidden_dims=tuple(raw_config["hidden_dims"]),
        learning_rate=float(raw_config["learning_rate"]),
        batch_size=int(raw_config["batch_size"]),
        weight_decay=float(raw_config["weight_decay"]),
    )

    checkpoints: dict[int, dict] = {}

    for horizon in HORIZONS:
        path = checkpoints_dir / f"mlp_{horizon}h.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint {path}. Run hyperparameter search first."
            )

        checkpoints[horizon] = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    return config, checkpoints


def evaluate_test_set(
    df: pd.DataFrame,
    results_dir: Path,
    device: torch.device,
) -> None:
    config, checkpoints = load_selected_run(
        results_dir,
        device,
    )

    metric_rows = []
    prediction_frames = []

    print("\nProtected 2025 test evaluation")
    print("------------------------------")

    for horizon in HORIZONS:
        splits = horizon_safe_split(df, horizon)
        test_df = splits["test"]

        checkpoint = checkpoints[horizon]

        X_test_raw = test_df[FEATURE_COLS]
        y_test = test_df[TARGET_COLS[horizon]].to_numpy(dtype=np.float64)

        x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float64)
        x_std = np.asarray(checkpoint["x_std"], dtype=np.float64)
        y_mean = float(checkpoint["y_mean"])
        y_std = float(checkpoint["y_std"])

        X_test = standardize(
            X_test_raw,
            x_mean,
            x_std,
        ).astype(np.float32)

        model = TemperatureMLP(
            input_dim=len(FEATURE_COLS),
            hidden_dims=config.hidden_dims,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])

        pred_std = predict_standardized(
            model,
            X_test,
            device=device,
        )
        mlp_pred = y_mean + y_std * pred_std

        persistence = PersistenceModel()
        persistence_pred = persistence.predict(X_test_raw)

        # OLS is fit exactly as in train.py: training period only.
        train_df = splits["train"]
        X_train_raw = train_df[FEATURE_COLS]
        y_train = train_df[TARGET_COLS[horizon]].to_numpy(dtype=np.float64)

        baseline_x_mean, baseline_x_std = fit_standardizer(X_train_raw)
        X_train = standardize(
            X_train_raw,
            baseline_x_mean,
            baseline_x_std,
        )
        X_test_ols = standardize(
            X_test_raw,
            baseline_x_mean,
            baseline_x_std,
        )

        ols = MultipleLinearRegression().fit(X_train, y_train)
        ols_pred = ols.predict(X_test_ols)

        predictions = {
            "Persistence": persistence_pred,
            "MultipleLinearRegression": ols_pred,
            "MLP": mlp_pred,
        }

        persistence_rmse = rmse(y_test, persistence_pred)

        for model_name, pred in predictions.items():
            metrics = compute_metrics(y_test, pred)

            metric_rows.append(
                {
                    "horizon_hours": horizon,
                    "split": "test",
                    "model": model_name,
                    "n_samples": len(y_test),
                    "mae_C": metrics["mae"],
                    "rmse_C": metrics["rmse"],
                    "bias_C": metrics["bias"],
                    "skill_vs_persistence": (
                        0.0
                        if model_name == "Persistence"
                        else 1.0 - metrics["rmse"] / persistence_rmse
                    ),
                }
            )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "timestamp": test_df["timestamp"].to_numpy(),
                    "valid_time": (
                        test_df["timestamp"]
                        + pd.to_timedelta(horizon, unit="h")
                    ).to_numpy(),
                    "horizon_hours": horizon,
                    "observed_temp_C": y_test,
                    "persistence_pred_C": persistence_pred,
                    "linear_pred_C": ols_pred,
                    "mlp_pred_C": mlp_pred,
                }
            )
        )

        print(
            f"{horizon:>2} h: "
            f"Persistence={rmse(y_test, persistence_pred):.4f}, "
            f"OLS={rmse(y_test, ols_pred):.4f}, "
            f"MLP={rmse(y_test, mlp_pred):.4f} C RMSE"
        )

    test_metrics = pd.DataFrame(metric_rows)
    test_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    test_metrics.to_csv(
        results_dir / "final_test_metrics.csv",
        index=False,
    )
    test_predictions.to_csv(
        results_dir / "final_test_predictions.csv",
        index=False,
    )

    print("\nFinal test metrics saved to:")
    print(results_dir / "final_test_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and train one MLP per temperature forecast horizon."
        )
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "Evaluate the protected 2025 holdout using an already-saved "
            "selected configuration. Does not perform hyperparameter search."
        ),
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )

    args = parser.parse_args()

    if args.max_epochs <= 0:
        raise ValueError("--max-epochs must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    print(f"Using device: {device}")

    df = load_model_ready(args.data)

    if args.evaluate_test:
        evaluate_test_set(
            df=df,
            results_dir=args.results_dir,
            device=device,
        )
        return

    best_config, best_results, search_df = staged_search(
        df=df,
        device=device,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
    )

    save_selected_run(
        results_dir=args.results_dir,
        config=best_config,
        results=best_results,
        search_df=search_df,
    )

    print("\nSaved validation-stage outputs in:")
    print(args.results_dir)
    print(
        "\nDo NOT evaluate the 2025 holdout while still changing the model. "
        "Once the design is frozen, run:\n"
        "    python train_mlp.py --evaluate-test"
    )


if __name__ == "__main__":
    main()
