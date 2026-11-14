"""
download_spatial_observations.py

Download and clean hourly ECCC observations for the spatial extension.

Final network
-------------
Target:
    Calgary International Airport (YYC)

Auxiliary stations:
    Bow Valley        - west
    Sundre A          - northwest
    Drumheller East   - east
    Claresholm        - south
    Lacombe CDA 2     - north

Years:
    2016--2025 by default

The script caches each monthly ECCC CSV so interrupted or repeated runs do not
redownload files that are already present.

Outputs
-------
data/raw/<station_slug>/monthly/*.csv
    Cached raw monthly ECCC downloads.

data/raw/<station_slug>_raw.csv
    Concatenated raw ECCC data for each station.

data/clean/<station_slug>_hourly.csv
    Standardized hourly data containing:
        timestamp
        temperature
        dew_point
        relative_humidity
        wind_direction_10deg
        wind_speed
        station_pressure

data/station_download_summary.csv
    Row counts, time coverage, and missingness percentages for each station.

Usage
-----
From spatial_extension/:

    python download_spatial_observations.py

Optional:
    python download_spatial_observations.py --start-year 2016 --end-year 2025
    python download_spatial_observations.py --force
"""

from __future__ import annotations

import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"

DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data/raw"
CLEAN_DIR = BASE_DIR / "data/clean"
SUMMARY_PATH = BASE_DIR / "data/station_download_summary.csv"

# station_id is the ECCC bulk-download ID, not the climate ID.
STATIONS = {
    "yyc": {
        "name": "CALGARY INTL A",
        "station_id": 50430,
        "climate_id": "3031092",
        "role": "target",
        "direction": "target",
        "distance_km": 0.7,
    },
    "bow_valley": {
        "name": "BOW VALLEY",
        "station_id": 10222,
        "climate_id": "3050778",
        "role": "auxiliary",
        "direction": "W",
        "distance_km": 73.4,
    },
    "sundre": {
        "name": "SUNDRE A",
        "station_id": 10105,
        "climate_id": "3026KNQ",
        "role": "auxiliary",
        "direction": "NW",
        "distance_km": 86.8,
    },
    "drumheller": {
        "name": "DRUMHELLER EAST",
        "station_id": 10708,
        "climate_id": "30221LG",
        "role": "auxiliary",
        "direction": "E",
        "distance_km": 100.1,
    },
    "claresholm": {
        "name": "CLARESHOLM",
        "station_id": 2224,
        "climate_id": "3031640",
        "role": "auxiliary",
        "direction": "S",
        "distance_km": 126.6,
    },
    "lacombe": {
        "name": "LACOMBE CDA 2",
        "station_id": 10906,
        "climate_id": "3023722",
        "role": "auxiliary",
        "direction": "N",
        "distance_km": 149.2,
    },
}

REQUIRED_OUTPUT_COLS = [
    "timestamp",
    "temperature",
    "dew_point",
    "relative_humidity",
    "wind_direction_10deg",
    "wind_speed",
    "station_pressure",
]


def normalize_column_name(name: str) -> str:
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


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    normalized = {
        normalize_column_name(c): c
        for c in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    if required:
        raise KeyError(
            "Could not find required column. Tried: "
            + ", ".join(candidates)
            + "\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )

    return None


def find_header_row(text: str) -> int:
    for i, line in enumerate(text.splitlines()):
        if "Date/Time" in line and "Temp" in line:
            return i
    raise ValueError("Could not locate the ECCC CSV header row.")


def parse_eccc_csv(content: bytes) -> pd.DataFrame:
    """
    Parse both current and legacy ECCC bulk-download layouts.
    """
    try:
        df = pd.read_csv(io.BytesIO(content))
        if any("Date/Time" in str(c) for c in df.columns):
            return df
    except Exception:
        pass

    text = content.decode("utf-8-sig", errors="replace")
    header_row = find_header_row(text)
    data_text = "\n".join(text.splitlines()[header_row:])
    return pd.read_csv(io.StringIO(data_text))


def validate_station(
    raw: pd.DataFrame,
    expected_climate_id: str,
    station_name: str,
) -> None:
    climate_col = find_column(
        raw,
        ["Climate ID"],
        required=False,
    )

    if climate_col is None:
        print(
            f"    warning: Climate ID column absent for {station_name}; "
            "identity could not be checked from the returned file."
        )
        return

    ids = (
        raw[climate_col]
        .dropna()
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .unique()
    )

    if len(ids) == 0:
        print(
            f"    warning: Climate ID field empty for {station_name}."
        )
        return

    if expected_climate_id not in ids:
        raise ValueError(
            f"{station_name}: expected Climate ID {expected_climate_id}, "
            f"but the download returned {ids.tolist()}."
        )


def station_month_dir(slug: str) -> Path:
    return RAW_DIR / slug / "monthly"


def download_month(
    session: requests.Session,
    slug: str,
    station_id: int,
    year: int,
    month: int,
    force: bool,
    pause_seconds: float,
) -> pd.DataFrame:
    monthly_dir = station_month_dir(slug)
    monthly_dir.mkdir(parents=True, exist_ok=True)

    cache_path = (
        monthly_dir
        / f"eccc_{station_id}_{year}_{month:02d}.csv"
    )

    if cache_path.exists() and not force:
        return pd.read_csv(cache_path)

    params = {
        "format": "csv",
        "stationID": station_id,
        "Year": year,
        "Month": month,
        "Day": 1,
        "timeframe": 1,
        "submit": "Download Data",
    }

    response = session.get(
        BASE_URL,
        params=params,
        timeout=60,
    )
    response.raise_for_status()

    df = parse_eccc_csv(response.content)

    if df.empty:
        raise ValueError(
            f"No hourly data returned for station {station_id}, "
            f"{year}-{month:02d}."
        )

    df.to_csv(cache_path, index=False)

    if pause_seconds > 0:
        time.sleep(pause_seconds)

    return df


def standardize_hourly_columns(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract the same physical variables used by the local experiment.
    """
    col_timestamp = find_column(
        raw,
        ["Date/Time (LST)", "Date/Time"],
    )
    col_temp = find_column(
        raw,
        ["Temp (°C)", "Temp (C)", "Temperature (°C)"],
    )
    col_dew = find_column(
        raw,
        ["Dew Point Temp (°C)", "Dew Point Temp (C)"],
    )
    col_rh = find_column(
        raw,
        ["Rel Hum (%)", "Relative Humidity (%)"],
    )
    col_wdir = find_column(
        raw,
        ["Wind Dir (10s deg)", "Wind Direction (10s deg)"],
    )
    col_wspd = find_column(
        raw,
        ["Wind Spd (km/h)", "Wind Speed (km/h)"],
    )
    col_press = find_column(
        raw,
        ["Stn Press (kPa)", "Station Pressure (kPa)"],
    )

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

    data.columns = REQUIRED_OUTPUT_COLS

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    numeric_cols = [
        c for c in REQUIRED_OUTPUT_COLS
        if c != "timestamp"
    ]

    for col in numeric_cols:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    data = (
        data
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return data


def complete_hourly_grid(
    clean: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Reindex onto an exact hourly grid.

    This preserves missing observations explicitly as NaN and guarantees
    later lag operations correspond to meteorological hours rather than rows.
    """
    start = pd.Timestamp(
        f"{start_year}-01-01 00:00:00"
    )
    end = pd.Timestamp(
        f"{end_year}-12-31 23:00:00"
    )

    full_index = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )

    grid = (
        clean
        .set_index("timestamp")
        .reindex(full_index)
    )

    grid.index.name = "timestamp"
    return grid.reset_index()


def download_station(
    session: requests.Session,
    slug: str,
    info: dict,
    start_year: int,
    end_year: int,
    force: bool,
    pause_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    name = info["name"]
    station_id = info["station_id"]

    print(
        f"\n{name} "
        f"(stationID={station_id}, "
        f"{info['direction']}, "
        f"{info['distance_km']:.1f} km)"
    )
    print("-" * max(40, len(name)))

    frames: list[pd.DataFrame] = []
    total = (end_year - start_year + 1) * 12
    k = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            k += 1
            print(
                f"  [{k:3d}/{total}] "
                f"{year}-{month:02d}",
                end=" ... ",
            )

            try:
                df = download_month(
                    session=session,
                    slug=slug,
                    station_id=station_id,
                    year=year,
                    month=month,
                    force=force,
                    pause_seconds=pause_seconds,
                )
                frames.append(df)
                print(f"{len(df)} rows")
            except Exception as exc:
                # Continue so one bad month produces missingness rather than
                # destroying the entire station audit.
                print(f"FAILED: {exc}")

    if not frames:
        raise RuntimeError(
            f"No data were downloaded for {name}."
        )

    raw = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    validate_station(
        raw,
        expected_climate_id=info["climate_id"],
        station_name=name,
    )

    raw_path = RAW_DIR / f"{slug}_raw.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)

    clean = standardize_hourly_columns(raw)
    clean = complete_hourly_grid(
        clean,
        start_year=start_year,
        end_year=end_year,
    )

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = CLEAN_DIR / f"{slug}_hourly.csv"
    clean.to_csv(clean_path, index=False)

    expected_rows = len(clean)

    summary = {
        "slug": slug,
        "station_name": name,
        "station_id": station_id,
        "climate_id": info["climate_id"],
        "role": info["role"],
        "direction": info["direction"],
        "distance_km": info["distance_km"],
        "expected_hourly_rows": expected_rows,
        "first_timestamp": clean["timestamp"].min(),
        "last_timestamp": clean["timestamp"].max(),
    }

    for col in REQUIRED_OUTPUT_COLS:
        if col == "timestamp":
            continue

        missing_n = int(clean[col].isna().sum())
        summary[f"{col}_missing_n"] = missing_n
        summary[f"{col}_missing_pct"] = (
            100.0 * missing_n / expected_rows
        )

    return clean, summary


def print_summary(summary_df: pd.DataFrame) -> None:
    display_cols = [
        "station_name",
        "direction",
        "distance_km",
        "temperature_missing_pct",
        "dew_point_missing_pct",
        "relative_humidity_missing_pct",
        "wind_speed_missing_pct",
        "station_pressure_missing_pct",
    ]

    display = summary_df[display_cols].copy()

    pct_cols = [
        c for c in display.columns
        if c.endswith("_missing_pct")
    ]

    display[pct_cols] = display[pct_cols].round(2)
    display["distance_km"] = display["distance_km"].round(1)

    print("\nDownload / missingness summary")
    print("------------------------------")
    print(display.to_string(index=False))

    print(f"\nSaved summary: {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download 2016--2025 hourly observations for the selected "
            "Calgary spatial forecasting network."
        )
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_START_YEAR,
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_END_YEAR,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload monthly files even when cached.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.10,
        help="Pause between uncached requests (default: 0.10 s).",
    )

    args = parser.parse_args()

    if args.end_year < args.start_year:
        raise ValueError(
            "--end-year must be greater than or equal to --start-year."
        )
    if args.pause_seconds < 0:
        raise ValueError("--pause-seconds cannot be negative.")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "calgary-spatial-temperature-forecast-project/1.0 "
                "(academic data-analysis project)"
            )
        }
    )

    summaries: list[dict] = []

    for slug, info in STATIONS.items():
        _, summary = download_station(
            session=session,
            slug=slug,
            info=info,
            start_year=args.start_year,
            end_year=args.end_year,
            force=args.force,
            pause_seconds=args.pause_seconds,
        )
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print_summary(summary_df)


if __name__ == "__main__":
    main()
