from __future__ import annotations

import argparse
import io
import time
from pathlib import Path


import numpy as np
import pandas as pd
import requests


# -----------------------------------------------------------------------------
# Project configuration
# -----------------------------------------------------------------------------

# Calgary International Airport (CALGARY INTL A)
# ECCC Climate ID: 3031092
# ECCC internal Station ID used by the bulk-download endpoint: 50430
STATION_ID = 50430
EXPECTED_CLIMATE_ID = "3031092"

DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025

HORIZONS = (1, 6, 12, 24, 48)
N_TEMP_LAGS = 24

BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data/raw"
MONTHLY_DIR = RAW_DIR / "monthly"
PROCESSED_DIR = BASE_DIR / "data/processed"

RAW_COMBINED_PATH = RAW_DIR / "eccc_calgary_hourly_raw.csv"
CLEAN_HOURLY_PATH = PROCESSED_DIR / "eccc_calgary_hourly_clean.csv"
MODEL_READY_PATH = PROCESSED_DIR / "model_ready.csv"
MISSINGNESS_PATH = PROCESSED_DIR / "missingness_report.csv"


# -----------------------------------------------------------------------------
# ECCC download helpers
# -----------------------------------------------------------------------------

def _find_header_row(text: str) -> int:
    """Locate the actual CSV header if ECCC prepends metadata lines."""
    for i, line in enumerate(text.splitlines()):
        if "Date/Time" in line and "Temp" in line:
            return i
    raise ValueError("Could not locate the ECCC hourly-data header row.")


def _parse_eccc_csv(content: bytes) -> pd.DataFrame:
    """Parse an ECCC CSV response, tolerating both current and legacy layouts."""
    try:
        df = pd.read_csv(io.BytesIO(content))
        if any("Date/Time" in str(c) for c in df.columns):
            return df
    except Exception:
        pass

    text = content.decode("utf-8-sig", errors="replace")
    header_row = _find_header_row(text)
    data_text = "\n".join(text.splitlines()[header_row:])
    return pd.read_csv(io.StringIO(data_text))


def download_month(
    session: requests.Session,
    year: int,
    month: int,
    force: bool = False,
    pause_seconds: float = 0.15,
) -> pd.DataFrame:
    """Download one month of hourly ECCC data, with local caching."""
    MONTHLY_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = MONTHLY_DIR / f"eccc_{STATION_ID}_{year}_{month:02d}.csv"

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    params = {
        "format": "csv",
        "stationID": STATION_ID,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,  # hourly
        "submit": "Download Data",
    }

    response = session.get(BASE_URL, params=params, timeout=60)
    response.raise_for_status()

    df = _parse_eccc_csv(response.content)
    if df.empty:
        raise ValueError(f"No data returned for {year}-{month:02d}.")

    df.to_csv(cache_path, index=False)
    time.sleep(pause_seconds)
    return df


def download_all_hourly_data(
    start_year: int,
    end_year: int,
    force: bool = False,
) -> pd.DataFrame:
    """Download and concatenate monthly hourly files for the requested years."""
    if end_year < start_year:
        raise ValueError("end_year must be greater than or equal to start_year.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "calgary-temperature-forecast-project/1.0 "
                "(academic data-analysis project)"
            )
        }
    )

    frames: list[pd.DataFrame] = []
    total = (end_year - start_year + 1) * 12
    count = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            count += 1
            print(f"[{count:3d}/{total}] {year}-{month:02d}", end=" ... ")
            try:
                df = download_month(session, year, month, force=force)
                frames.append(df)
                print(f"{len(df)} rows")
            except Exception as exc:
                print(f"FAILED: {exc}")
                raise

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(RAW_COMBINED_PATH, index=False)
    return combined


# -----------------------------------------------------------------------------
# Column handling
# -----------------------------------------------------------------------------

def _normalize_column_name(name: str) -> str:
    s = str(name).strip().lower()
    replacements = {
        "°": "",
        "(": "",
        ")": "",
        "/": " ",
        "-": " ",
        ".": "",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return " ".join(s.split())


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {_normalize_column_name(c): c for c in df.columns}

    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    raise KeyError(
        "Could not find a required column. Tried: "
        + ", ".join(candidates)
        + "\nAvailable columns:\n"
        + "\n".join(map(str, df.columns))
    )


def select_and_standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Select the meteorological variables used by the forecasting project."""
    col_timestamp = _find_column(raw, ["Date/Time (LST)", "Date/Time"])
    col_temp = _find_column(raw, ["Temp (°C)", "Temp (C)", "Temperature (°C)"])
    col_dew = _find_column(raw, ["Dew Point Temp (°C)", "Dew Point Temp (C)"])
    col_rh = _find_column(raw, ["Rel Hum (%)", "Relative Humidity (%)"])
    col_wdir = _find_column(raw, ["Wind Dir (10s deg)", "Wind Direction (10s deg)"])
    col_wspd = _find_column(raw, ["Wind Spd (km/h)", "Wind Speed (km/h)"])
    col_press = _find_column(raw, ["Stn Press (kPa)", "Station Pressure (kPa)"])

    data = raw[
        [
            col_timestamp,
            col_temp,
            col_dew,
            col_rh,
            col_wdir,
            col_wspd,
            col_press,
        ]
    ].copy()

    data.columns = [
        "timestamp",
        "temperature",
        "dew_point",
        "relative_humidity",
        "wind_direction_10deg",
        "wind_speed",
        "station_pressure",
    ]

    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    numeric_cols = [
        "temperature",
        "dew_point",
        "relative_humidity",
        "wind_direction_10deg",
        "wind_speed",
        "station_pressure",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["timestamp"])
    data = data.sort_values("timestamp")
    data = data.drop_duplicates(subset="timestamp", keep="last")

    return data


def validate_station(raw: pd.DataFrame) -> None:
    """Check that the download corresponds to the intended Calgary station."""
    try:
        climate_col = _find_column(raw, ["Climate ID"])
    except KeyError:
        print("Warning: Climate ID column not found; station identity was not validated.")
        return

    ids = (
        raw[climate_col]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .unique()
    )

    if len(ids) == 0:
        print("Warning: Climate ID is empty; station identity was not validated.")
        return

    if EXPECTED_CLIMATE_ID not in ids:
        raise ValueError(
            f"Expected Climate ID {EXPECTED_CLIMATE_ID}, but downloaded IDs were {ids}. "
            "Check STATION_ID before continuing."
        )


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

def create_complete_hourly_grid(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex onto a complete hourly LST grid.

    This is essential: after reindexing, shift(24) really means 24 hours,
    rather than merely 24 observed rows.
    """
    hourly = clean.set_index("timestamp").sort_index()

    full_index = pd.date_range(
        start=hourly.index.min(),
        end=hourly.index.max(),
        freq="h",
        name="timestamp",
    )

    hourly = hourly.reindex(full_index)
    return hourly


def add_wind_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode circular wind direction as sine/cosine components."""
    out = df.copy()

    # ECCC hourly wind direction is reported in tens of degrees.
    direction_deg = 10.0 * out["wind_direction_10deg"]
    direction_rad = np.deg2rad(direction_deg)

    out["wind_dir_sin"] = np.sin(direction_rad)
    out["wind_dir_cos"] = np.cos(direction_rad)

    # For calm conditions ECCC may leave direction blank. A zero vector is a
    # natural representation of "no wind direction" when wind speed is zero.
    calm = out["wind_speed"].eq(0) & out["wind_direction_10deg"].isna()
    out.loc[calm, ["wind_dir_sin", "wind_dir_cos"]] = 0.0

    return out


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour-of-day and day-of-year features."""
    out = df.copy()

    hour = out.index.hour.to_numpy()
    day_of_year = out.index.dayofyear.to_numpy()

    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    # 365.25 is adequate here; this is a smooth seasonality feature, not a
    # calendar identity variable.
    out["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)

    return out


def build_model_ready_dataset(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create one row per forecast origin time.

    Features:
      - temperature at t, t-1, ..., t-23
      - current dew point, relative humidity, pressure, wind speed
      - current wind-direction sine/cosine
      - hour-of-day sine/cosine
      - day-of-year sine/cosine

    Targets:
      - temperature at t+1, t+6, t+12, t+24, t+48
    """
    df = add_wind_direction_features(hourly)
    df = add_calendar_features(df)

    model = pd.DataFrame(index=df.index)

    for lag in range(N_TEMP_LAGS):
        model[f"temp_lag_{lag}"] = df["temperature"].shift(lag)

    model["dew_point"] = df["dew_point"]
    model["relative_humidity"] = df["relative_humidity"]
    model["station_pressure"] = df["station_pressure"]
    model["wind_speed"] = df["wind_speed"]
    model["wind_dir_sin"] = df["wind_dir_sin"]
    model["wind_dir_cos"] = df["wind_dir_cos"]
    model["hour_sin"] = df["hour_sin"]
    model["hour_cos"] = df["hour_cos"]
    model["doy_sin"] = df["doy_sin"]
    model["doy_cos"] = df["doy_cos"]

    for horizon in HORIZONS:
        model[f"target_temp_{horizon}h"] = df["temperature"].shift(-horizon)

    feature_cols = [f"temp_lag_{lag}" for lag in range(N_TEMP_LAGS)] + [
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
    target_cols = [f"target_temp_{h}h" for h in HORIZONS]
    required_cols = feature_cols + target_cols

    missingness = pd.DataFrame(
        {
            "missing_count": model[required_cols].isna().sum(),
            "missing_fraction": model[required_cols].isna().mean(),
        }
    )
    missingness.index.name = "column"

    # Common complete-case dataset so all forecast horizons are evaluated on
    # the same set of forecast-origin times.
    model_ready = model.dropna(subset=required_cols).copy()
    model_ready.index.name = "timestamp"
    model_ready = model_ready.reset_index()

    return model_ready, missingness


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    validate_station(raw)
    clean = select_and_standardize_columns(raw)
    hourly = create_complete_hourly_grid(clean)

    hourly.reset_index().to_csv(CLEAN_HOURLY_PATH, index=False)

    model_ready, missingness = build_model_ready_dataset(hourly)
    model_ready.to_csv(MODEL_READY_PATH, index=False)
    missingness.to_csv(MISSINGNESS_PATH)

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Observed rows after duplicate removal: {len(clean):,}")
    print(f"Rows on complete hourly grid:          {len(hourly):,}")
    print(f"Model-ready complete rows:             {len(model_ready):,}")
    print(f"Input features:                        34")
    print(f"Forecast horizons:                     {HORIZONS}")
    print(f"\nSaved cleaned hourly data: {CLEAN_HOURLY_PATH}")
    print(f"Saved model-ready data:   {MODEL_READY_PATH}")
    print(f"Saved missingness report: {MISSINGNESS_PATH}")

    return model_ready


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download ECCC hourly observations for Calgary International Airport "
            "and build a PyTorch-friendly temperature-forecasting CSV."
        )
    )
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument("--end-year", type=int, default=DEFAULT_END_YEAR)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload monthly files even if cached locally.",
    )
    args = parser.parse_args()

    raw = download_all_hourly_data(
        start_year=args.start_year,
        end_year=args.end_year,
        force=args.force_download,
    )
    preprocess(raw)


if __name__ == "__main__":
    main()
