"""
download_spatial_data.py

Phase 1 of the spatial extension:
    Audit ECCC climate stations around Calgary International Airport (YYC)
    and identify stations with hourly coverage suitable for 2016--2025.

This script does NOT yet download the full 10-year observation archive for all
stations. It first creates a defensible station shortlist.

Outputs
-------
data/station_audit_all.csv
    All ECCC climate stations within the requested radius that expose hourly
    station metadata.

data/station_audit_full_coverage.csv
    Subset whose hourly record begins on/before the requested start date and
    extends through the requested end date.

data/station_audit_directional_shortlist.csv
    A small automatically selected shortlist intended to provide directional
    coverage around Calgary (W, E, N, S plus a near-field station when possible).

Usage
-----
From spatial_extension/:

    python download_spatial_data.py

Optional:

    python download_spatial_data.py --radius-km 150
    python download_spatial_data.py --start-year 2016 --end-year 2025
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Calgary International Airport target
# ---------------------------------------------------------------------------

TARGET_NAME = "CALGARY INTL A"
TARGET_LAT = 51.1167
TARGET_LON = -114.0167

DEFAULT_RADIUS_KM = 150.0
DEFAULT_START_YEAR = 2016
DEFAULT_END_YEAR = 2025

STATION_API = (
    "https://api.weather.gc.ca/collections/climate-stations/items"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ALL_CANDIDATES_PATH = DATA_DIR / "station_audit_all.csv"
FULL_COVERAGE_PATH = DATA_DIR / "station_audit_full_coverage.csv"
SHORTLIST_PATH = DATA_DIR / "station_audit_directional_shortlist.csv"


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two latitude/longitude points."""
    radius_earth_km = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2.0) ** 2
    )

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_earth_km * c


def initial_bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Initial bearing from point 1 to point 2 in degrees clockwise from north.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    y = math.sin(dlambda) * math.cos(phi2)
    x = (
        math.cos(phi1) * math.sin(phi2)
        - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    )

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360.0) % 360.0


def compass_sector(bearing_deg: float) -> str:
    """Convert bearing to an 8-sector compass direction."""
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = int((bearing_deg + 22.5) // 45.0) % 8
    return labels[index]


# ---------------------------------------------------------------------------
# ECCC station metadata retrieval
# ---------------------------------------------------------------------------

def bbox_for_radius(
    lat: float,
    lon: float,
    radius_km: float,
) -> tuple[float, float, float, float]:
    """
    Construct a conservative rectangular bounding box around the target.

    The final radius filter is applied using the haversine distance, so this
    box only needs to be large enough to avoid missing stations.
    """
    lat_delta = radius_km / 111.0
    lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))

    return (
        lon - lon_delta,
        lat - lat_delta,
        lon + lon_delta,
        lat + lat_delta,
    )


def fetch_station_features(
    radius_km: float,
) -> list[dict[str, Any]]:
    """
    Query the ECCC OGC API for climate-station metadata near Calgary.

    The API is a standard OGC Features endpoint and supports bbox queries.
    """
    bbox = bbox_for_radius(TARGET_LAT, TARGET_LON, radius_km)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "calgary-spatial-temperature-forecast-project/1.0 "
                "(academic data-analysis project)"
            )
        }
    )

    params = {
        "f": "json",
        "bbox": ",".join(f"{x:.6f}" for x in bbox),
        "limit": 1000,
    }

    response = session.get(STATION_API, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    features = payload.get("features", [])
    if not features:
        raise RuntimeError(
            "ECCC station API returned no features. "
            "Check internet access or the API endpoint."
        )

    # In practice 1000 is far more than enough for a 150 km box, but support
    # OGC pagination in case ECCC returns a next link.
    seen_urls = {response.url}
    links = payload.get("links", [])

    while True:
        next_url = None

        for link in links:
            if link.get("rel") == "next" and link.get("href"):
                next_url = link["href"]
                break

        if not next_url or next_url in seen_urls:
            break

        seen_urls.add(next_url)
        response = session.get(next_url, timeout=60)
        response.raise_for_status()
        payload = response.json()

        features.extend(payload.get("features", []))
        links = payload.get("links", [])

    return features


def parse_datetime(value: Any) -> pd.Timestamp:
    if value is None or value == "":
        return pd.NaT
    return pd.to_datetime(value, errors="coerce")


def features_to_dataframe(
    features: list[dict[str, Any]],
    radius_km: float,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    start_required = pd.Timestamp(f"{start_year}-01-01 00:00:00")
    end_required = pd.Timestamp(f"{end_year}-12-31 23:00:00")

    rows: list[dict[str, Any]] = []

    for feature in features:
        props = feature.get("properties", {}) or {}
        geometry = feature.get("geometry", {}) or {}

        coords = geometry.get("coordinates")
        if not coords or len(coords) < 2:
            continue

        lon = float(coords[0])
        lat = float(coords[1])

        distance = haversine_km(
            TARGET_LAT,
            TARGET_LON,
            lat,
            lon,
        )

        if distance > radius_km:
            continue

        hly_first = parse_datetime(props.get("HLY_FIRST_DATE"))
        hly_last = parse_datetime(props.get("HLY_LAST_DATE"))

        hourly_flag = str(
            props.get("HAS_HOURLY_DATA", "")
        ).strip().upper()

        has_hourly_metadata = (
            pd.notna(hly_first)
            and pd.notna(hly_last)
        )

        # Some ECCC metadata records can have inconsistent legacy flags, so
        # date coverage is the main criterion. Keep the flag for diagnosis.
        full_coverage = bool(
            has_hourly_metadata
            and hly_first <= start_required
            and hly_last >= end_required
        )

        bearing = initial_bearing_deg(
            TARGET_LAT,
            TARGET_LON,
            lat,
            lon,
        )

        rows.append(
            {
                "climate_id": (
                    props.get("CLIMATE_IDENTIFIER")
                    or feature.get("id")
                ),
                "station_id": props.get("STN_ID"),
                "station_name": props.get("STATION_NAME"),
                "tc_identifier": props.get("TC_IDENTIFIER"),
                "wmo_identifier": props.get("WMO_IDENTIFIER"),
                "station_type": props.get("STATION_TYPE"),
                "latitude": lat,
                "longitude": lon,
                "elevation_m": pd.to_numeric(
                    props.get("ELEVATION"),
                    errors="coerce",
                ),
                "distance_km": distance,
                "bearing_deg": bearing,
                "direction": compass_sector(bearing),
                "has_hourly_flag": hourly_flag,
                "hourly_first": hly_first,
                "hourly_last": hly_last,
                "full_coverage": full_coverage,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No stations remained after the radius filter."
        )

    df = df.sort_values(
        ["full_coverage", "distance_km"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Shortlisting
# ---------------------------------------------------------------------------

def choose_directional_shortlist(
    full: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce an automatic, conservative shortlist.

    Selection logic:
      1. one near-field station (excluding the YYC target itself);
      2. one western station;
      3. one eastern station;
      4. one northern station;
      5. one southern station.

    Directional sectors are grouped broadly because the exact final station
    set should be reviewed manually before the large download.
    """
    if full.empty:
        return full.copy()

    work = full.copy()

    # Exclude the target or effectively co-located duplicates.
    work = work[
        ~(
            work["station_name"]
            .fillna("")
            .str.upper()
            .str.contains("CALGARY INTL")
        )
    ].copy()

    work = work[work["distance_km"] >= 5.0].copy()

    selected_indices: list[int] = []

    def take_best(mask: pd.Series) -> None:
        candidates = work.loc[mask].copy()
        candidates = candidates[
            ~candidates.index.isin(selected_indices)
        ]

        if candidates.empty:
            return

        # Prefer nearer stations, but gently reward elevation contrast so that
        # a station is not selected solely because it is geographically close.
        target_elevation_guess = 1084.0
        candidates["selection_score"] = (
            candidates["distance_km"]
            - 0.015
            * (
                candidates["elevation_m"]
                - target_elevation_guess
            ).abs().fillna(0.0)
        )

        best_idx = candidates["selection_score"].idxmin()
        selected_indices.append(best_idx)

    # One near station first.
    near = work[work["distance_km"] <= 40.0]
    if not near.empty:
        selected_indices.append(
            near["distance_km"].idxmin()
        )

    # Broad directional coverage.
    take_best(work["direction"].isin(["W", "NW", "SW"]))
    take_best(work["direction"].isin(["E", "NE", "SE"]))
    take_best(work["direction"].isin(["N", "NE", "NW"]))
    take_best(work["direction"].isin(["S", "SE", "SW"]))

    # Deduplicate while retaining order.
    selected_indices = list(dict.fromkeys(selected_indices))

    shortlist = work.loc[selected_indices].copy()
    shortlist = shortlist.sort_values("distance_km").reset_index(drop=True)

    return shortlist


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(
    all_candidates: pd.DataFrame,
    full: pd.DataFrame,
    shortlist: pd.DataFrame,
    radius_km: float,
    start_year: int,
    end_year: int,
) -> None:
    print("\nSpatial station audit")
    print("---------------------")
    print(f"Target:               {TARGET_NAME}")
    print(f"Radius:               {radius_km:.0f} km")
    print(f"Required hourly span: {start_year}-01-01 through {end_year}-12-31")
    print(f"Stations in radius:   {len(all_candidates)}")
    print(f"Full-span candidates: {len(full)}")

    columns = [
        "station_name",
        "station_id",
        "climate_id",
        "distance_km",
        "direction",
        "elevation_m",
        "hourly_first",
        "hourly_last",
    ]

    if not shortlist.empty:
        display = shortlist[columns].copy()
        display["distance_km"] = display["distance_km"].round(1)
        display["elevation_m"] = display["elevation_m"].round(1)

        print("\nAutomatic directional shortlist")
        print("--------------------------------")
        print(display.to_string(index=False))
    else:
        print(
            "\nNo station met the full-span criteria. "
            "We may need to thread predecessor/successor stations."
        )

    print("\nSaved:")
    print(f"  {ALL_CANDIDATES_PATH}")
    print(f"  {FULL_COVERAGE_PATH}")
    print(f"  {SHORTLIST_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit ECCC hourly climate stations around Calgary for the "
            "multi-station forecasting extension."
        )
    )

    parser.add_argument(
        "--radius-km",
        type=float,
        default=DEFAULT_RADIUS_KM,
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

    args = parser.parse_args()

    if args.radius_km <= 0:
        raise ValueError("--radius-km must be positive.")
    if args.end_year < args.start_year:
        raise ValueError("--end-year must be >= --start-year.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Querying ECCC climate-station metadata ...")
    features = fetch_station_features(args.radius_km)

    all_candidates = features_to_dataframe(
        features=features,
        radius_km=args.radius_km,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    full = all_candidates[
        all_candidates["full_coverage"]
    ].copy()

    shortlist = choose_directional_shortlist(full)

    all_candidates.to_csv(
        ALL_CANDIDATES_PATH,
        index=False,
    )
    full.to_csv(
        FULL_COVERAGE_PATH,
        index=False,
    )
    shortlist.to_csv(
        SHORTLIST_PATH,
        index=False,
    )

    print_summary(
        all_candidates=all_candidates,
        full=full,
        shortlist=shortlist,
        radius_km=args.radius_km,
        start_year=args.start_year,
        end_year=args.end_year,
    )


if __name__ == "__main__":
    main()
