"""
preprocess_spatial.py

Merge the cleaned hourly station files into a spatial supervised-learning
dataset for Calgary temperature forecasting.

Design
------
The matched experiment uses 34 Calgary/YYC predictors. It requires a reported
wind direction; the separate original local experiment also retains calm hours:

    24 hourly YYC temperature lags: t, t-1, ..., t-23
    current YYC dew point
    current YYC relative humidity
    current YYC station pressure
    current YYC wind speed
    YYC wind direction as sine/cosine
    hour of day as sine/cosine
    day of year as sine/cosine

Spatial extension:
    For each of five auxiliary stations:
        temperature at t, t-3, t-6
        current dew point
        current relative humidity
        current wind speed

    Current station pressure is additionally retained for:
        Bow Valley
        Sundre A
        Claresholm

    It is excluded at Drumheller East and Lacombe CDA 2 because the station
    audit found substantial pressure missingness there.

Auxiliary missing values are intentionally preserved as NaN here.
They must be imputed later using TRAINING-SET statistics only.

Targets
-------
YYC temperature at:
    +1 h, +6 h, +12 h, +24 h, +48 h

Outputs
-------
data/processed/model_ready_spatial.csv
data/processed/spatial_feature_missingness.csv
data/processed/spatial_feature_manifest.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data/clean"
PROCESSED_DIR = BASE_DIR / "data/processed"

OUTPUT_PATH = PROCESSED_DIR / "model_ready_spatial.csv"
MISSINGNESS_PATH = PROCESSED_DIR / "spatial_feature_missingness.csv"
MANIFEST_PATH = PROCESSED_DIR / "spatial_feature_manifest.csv"

HORIZONS = (1, 6, 12, 24, 48)

AUXILIARY_STATIONS = {
    "bow_valley": {
        "label": "Bow Valley",
        "include_pressure": True,
    },
    "sundre": {
        "label": "Sundre A",
        "include_pressure": True,
    },
    "drumheller": {
        "label": "Drumheller East",
        "include_pressure": False,
    },
    "claresholm": {
        "label": "Claresholm",
        "include_pressure": True,
    },
    "lacombe": {
        "label": "Lacombe CDA 2",
        "include_pressure": False,
    },
}


def load_clean_station(slug: str) -> pd.DataFrame:
    path = CLEAN_DIR / f"{slug}_hourly.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing cleaned station file: {path}\n"
            "Run download_spatial_observations.py first."
        )

    df = pd.read_csv(path)

    required = {
        "timestamp",
        "temperature",
        "dew_point",
        "relative_humidity",
        "wind_direction_10deg",
        "wind_speed",
        "station_pressure",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            f"{path} contains invalid timestamps."
        )

    if df["timestamp"].duplicated().any():
        raise ValueError(
            f"{path} contains duplicate timestamps."
        )

    df = df.sort_values("timestamp").reset_index(drop=True)
    if df.empty or not df["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError(f"{path} must have a continuous hourly grid before shifting rows.")
    return df


def ensure_same_grid(
    reference: pd.DataFrame,
    other: pd.DataFrame,
    slug: str,
) -> None:
    if len(reference) != len(other):
        raise ValueError(
            f"{slug}: hourly grid length differs from YYC."
        )

    if not reference["timestamp"].equals(other["timestamp"]):
        raise ValueError(
            f"{slug}: timestamps do not align exactly with YYC."
        )


def add_local_features(
    yyc: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    out = pd.DataFrame(
        {"timestamp": yyc["timestamp"]}
    )

    manifest: list[dict] = []

    # 24 hourly temperature lags, including current temperature.
    for lag in range(24):
        col = f"temp_lag_{lag}"
        out[col] = yyc["temperature"].shift(lag)

        manifest.append(
            {
                "feature": col,
                "source_station": "CALGARY INTL A",
                "feature_group": "local",
                "variable": "temperature",
                "lag_hours": lag,
            }
        )

    local_current = {
        "dew_point": "dew_point",
        "relative_humidity": "relative_humidity",
        "station_pressure": "station_pressure",
        "wind_speed": "wind_speed",
    }

    for output_col, source_col in local_current.items():
        out[output_col] = yyc[source_col]

        manifest.append(
            {
                "feature": output_col,
                "source_station": "CALGARY INTL A",
                "feature_group": "local",
                "variable": source_col,
                "lag_hours": 0,
            }
        )

    # Preserve the matched experiment's reported-direction cohort. Blank calm
    # directions stay missing here. verify_results.py also scores the zero-vector encoding.
    # Wind direction in the ECCC hourly archive is recorded in tens of degrees.
    theta_rad = np.deg2rad(
        10.0 * yyc["wind_direction_10deg"]
    )

    out["wind_dir_sin"] = np.sin(theta_rad)
    out["wind_dir_cos"] = np.cos(theta_rad)

    for col in ("wind_dir_sin", "wind_dir_cos"):
        manifest.append(
            {
                "feature": col,
                "source_station": "CALGARY INTL A",
                "feature_group": "local",
                "variable": "wind_direction",
                "lag_hours": 0,
            }
        )

    hour = yyc["timestamp"].dt.hour.astype(float)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    day = yyc["timestamp"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(
        2.0 * np.pi * day / 365.25
    )
    out["doy_cos"] = np.cos(
        2.0 * np.pi * day / 365.25
    )

    for col in (
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ):
        manifest.append(
            {
                "feature": col,
                "source_station": "calendar",
                "feature_group": "local",
                "variable": col,
                "lag_hours": 0,
            }
        )

    return out, manifest


def add_spatial_features(
    out: pd.DataFrame,
    yyc: pd.DataFrame,
) -> list[dict]:
    manifest: list[dict] = []

    for slug, config in AUXILIARY_STATIONS.items():
        station = load_clean_station(slug)
        ensure_same_grid(yyc, station, slug)

        label = config["label"]

        # Selected temperature history.
        for lag in (0, 3, 6):
            col = f"{slug}_temp_lag_{lag}"
            out[col] = station["temperature"].shift(lag)

            manifest.append(
                {
                    "feature": col,
                    "source_station": label,
                    "feature_group": "spatial",
                    "variable": "temperature",
                    "lag_hours": lag,
                }
            )

        current_variables = {
            f"{slug}_dew_point": "dew_point",
            f"{slug}_relative_humidity": "relative_humidity",
            f"{slug}_wind_speed": "wind_speed",
        }

        if config["include_pressure"]:
            current_variables[
                f"{slug}_station_pressure"
            ] = "station_pressure"

        for output_col, source_col in current_variables.items():
            out[output_col] = station[source_col]

            manifest.append(
                {
                    "feature": output_col,
                    "source_station": label,
                    "feature_group": "spatial",
                    "variable": source_col,
                    "lag_hours": 0,
                }
            )

    return manifest


def add_targets(
    out: pd.DataFrame,
    yyc: pd.DataFrame,
) -> None:
    for horizon in HORIZONS:
        out[f"target_temp_{horizon}h"] = (
            yyc["temperature"].shift(-horizon)
        )


def write_missingness_report(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    rows = []

    for col in feature_cols:
        missing_n = int(df[col].isna().sum())
        rows.append(
            {
                "feature": col,
                "missing_n": missing_n,
                "missing_pct": (
                    100.0 * missing_n / len(df)
                ),
            }
        )

    report = (
        pd.DataFrame(rows)
        .sort_values(
            ["missing_pct", "feature"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )

    report.to_csv(
        MISSINGNESS_PATH,
        index=False,
    )


def main() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    yyc = load_clean_station("yyc")

    out, local_manifest = add_local_features(yyc)
    spatial_manifest = add_spatial_features(
        out=out,
        yyc=yyc,
    )

    add_targets(out, yyc)

    local_feature_cols = [
        row["feature"] for row in local_manifest
    ]
    spatial_feature_cols = [
        row["feature"] for row in spatial_manifest
    ]
    feature_cols = (
        local_feature_cols + spatial_feature_cols
    )

    target_cols = [
        f"target_temp_{h}h"
        for h in HORIZONS
    ]

    if len(local_feature_cols) != 34:
        raise AssertionError(
            f"Expected 34 local features, got "
            f"{len(local_feature_cols)}."
        )

    if len(spatial_feature_cols) != 33:
        raise AssertionError(
            f"Expected 33 auxiliary spatial features, got "
            f"{len(spatial_feature_cols)}."
        )

    if len(feature_cols) != 67:
        raise AssertionError(
            f"Expected 67 total features, got "
            f"{len(feature_cols)}."
        )

    # For a fair local-vs-spatial comparison, keep only rows for which the
    # ORIGINAL local predictor set and all five targets are available.
    #
    # Auxiliary missingness is preserved and will be handled using training
    # statistics in the model-training scripts.
    before = len(out)

    out = out.dropna(
        subset=local_feature_cols + target_cols
    ).reset_index(drop=True)

    after = len(out)

    ordered_cols = (
        ["timestamp"]
        + feature_cols
        + target_cols
    )
    out = out[ordered_cols]

    out.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    write_missingness_report(
        out,
        feature_cols=spatial_feature_cols,
    )

    manifest = pd.DataFrame(
        local_manifest + spatial_manifest
    )
    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("\nSpatial preprocessing complete")
    print("------------------------------")
    print(f"Hourly source rows:       {before:,}")
    print(f"Supervised rows retained: {after:,}")
    print(f"Local features:           {len(local_feature_cols)}")
    print(f"Spatial features:         {len(spatial_feature_cols)}")
    print(f"Total predictors:         {len(feature_cols)}")
    print(f"Targets:                  {len(target_cols)}")
    print()
    print("Auxiliary NaNs were NOT imputed.")
    print(
        "Training scripts must fit imputation values on "
        "the training period only."
    )
    print()
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {MISSINGNESS_PATH}")
    print(f"Saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
