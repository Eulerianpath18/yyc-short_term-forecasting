"""Build the portable NumPy bundle from this project's fitted checkpoints and scalers."""
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from architecture_extension.sequence_data import BASE_VARS, STATIONS
from predict import BUNDLE, HORIZONS, SEEDS

ROOT = Path(__file__).resolve().parent
SPATIAL = ROOT / "spatial_extension"
SEQUENCE = ROOT / "architecture_extension/results"
UNITS = {
    "temperature": "deg C", "dew_point": "deg C", "relative_humidity": "%",
    "wind_speed": "km/h", "station_pressure": "kPa", "wind_direction": "unitless",
    "hour_sin": "unitless", "hour_cos": "unitless", "doy_sin": "unitless", "doy_cos": "unitless",
}


def export(read_checkpoint, target=BUNDLE):
    manifest = pd.read_csv(SPATIAL / "data/processed/spatial_feature_manifest.csv")
    names = manifest.feature.tolist()
    local = manifest.loc[manifest.feature_group == "local", "feature"].tolist()
    if names[:len(local)] != local or (len(local), len(names)) != (34, 67):
        raise ValueError("Unexpected engineered feature schema.")
    bundle = {
        "feature_names": np.array(names),
        "feature_units": np.array([UNITS[v] for v in manifest.variable]),
        "auxiliary": (manifest.feature_group == "spatial").to_numpy(),
        "seq_channels": np.array([f"{station}__{v}" for station in STATIONS for v in BASE_VARS]),
        "seq_units": np.array([UNITS[v] for _ in STATIONS for v in BASE_VARS]),
    }

    stats = pd.read_csv(SPATIAL / "results/spatial_mlp/preprocessing_stats.csv")
    stats = stats.set_index(["horizon_hours", "model", "feature"])
    design = pd.read_csv(
        SPATIAL / "data/processed/model_ready_spatial.csv",
        parse_dates=["timestamp"], usecols=["timestamp", *[f"target_temp_{h}h" for h in HORIZONS]],
    )
    for horizon in HORIZONS:
        # Matched checkpoints store model tensors. Their target scale comes from the same lead's training rows.
        train_mask = design.timestamp + pd.Timedelta(hours=horizon) < "2024-01-01"
        targets = design.loc[train_mask, f"target_temp_{horizon}h"].to_numpy()
        for model in ("standalone-local", "local", "spatial"):
            key = f"{model}__{horizon}__"
            if model == "standalone-local":
                checkpoint = read_checkpoint(ROOT / f"results/mlp/checkpoints/mlp_{horizon}h.pt")
                if list(checkpoint["feature_cols"]) != local:
                    raise ValueError("Standalone checkpoint feature order differs from the manifest.")
                scalers = {
                    "median": np.zeros(len(local)),  # Local features are required and never imputed.
                    "mean": checkpoint["x_mean"], "std": checkpoint["x_std"],
                    "y_mean": checkpoint["y_mean"], "y_std": checkpoint["y_std"],
                }
            else:
                path = SPATIAL / f"results/spatial_mlp/checkpoints/{model}_mlp_{horizon}h.pt"
                checkpoint = read_checkpoint(path)
                features = local if model == "local" else names
                ordered = stats.loc[(horizon, model.title() + " MLP"), :].loc[features]
                scalers = {
                    "median": ordered.train_median, "mean": ordered.train_mean_after_imputation,
                    "std": ordered.train_std_after_imputation, "y_mean": targets.mean(),
                    "y_std": targets.std(ddof=0) or 1.0,
                }
            state = checkpoint["model_state_dict"]
            layers = sorted(int(n.split(".")[1]) for n in state if n.startswith("network.") and n.endswith(".weight"))
            for index, layer in enumerate(layers):
                bundle[f"{key}w{index}"] = np.asarray(state[f"network.{layer}.weight"])
                bundle[f"{key}b{index}"] = np.asarray(state[f"network.{layer}.bias"])
            bundle[key + "layers"] = np.asarray(len(layers))
            bundle.update({key + name: np.asarray(value) for name, value in scalers.items()})

    with np.load(SEQUENCE / "sequence_preprocessing_stats.npz", allow_pickle=False) as preprocessing:
        for name in ("median", "mean", "std"):
            bundle["seq_" + name] = preprocessing[name]
    for model in ("mlp", "tcn"):
        for seed in SEEDS:
            checkpoint = read_checkpoint(SEQUENCE / f"checkpoints/{model}_seed{seed}.pt")
            key = f"sequence-{model}__{seed}__"
            for name, value in checkpoint["state_dict"].items():
                bundle[key + name] = np.asarray(value)
            bundle[key + "y_mean"] = np.asarray(checkpoint["target_mean"])
            bundle[key + "y_std"] = np.asarray(checkpoint["target_std"])

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        np.savez_compressed(temporary, **bundle)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    import torch

    # The project's legacy checkpoints include NumPy arrays alongside their PyTorch state dictionaries.
    export(lambda path: torch.load(path, map_location="cpu", weights_only=False))
    print(f"Exported inference parameters to {BUNDLE}")
