"""Benchmark archived ECCC forecasts against YYC station observations.

Important time convention
-------------------------
The cleaned YYC observation file in this repository uses ECCC's Date/Time
(LST), i.e. fixed local standard time.  For Calgary this is MST = UTC-7
throughout the year; it does *not* switch to MDT in summer.

Forecast origins are supplied in UTC.  We therefore compute forecast valid
UTC and then convert to YYC LST by subtracting seven hours before matching to
the station observations.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "spatial_extension" / "data" / "clean" / "yyc_hourly.csv"
OUT = Path(__file__).resolve().parent / "results"
CALGARY_LST_UTC_OFFSET_HOURS = -7
ALLOWED_LEADS = {1, 6, 12, 24, 48}


def _parse_utc(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, utc=True, errors="raise")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score timestamp-matched HRDPS/PROGNOS forecasts at YYC."
    )
    ap.add_argument("forecast_csv")
    ap.add_argument("--output-dir", type=Path, default=OUT)
    args = ap.parse_args()

    f = pd.read_csv(args.forecast_csv)
    required = {"origin_utc", "lead_hours", "forecast_temp_C", "system"}
    missing = required - set(f.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    f["origin_utc"] = _parse_utc(f["origin_utc"])
    leads = pd.to_numeric(f["lead_hours"], errors="raise")
    if not leads.isin(ALLOWED_LEADS).all():
        raise ValueError(f"Forecast leads must be one of {sorted(ALLOWED_LEADS)} hours.")
    f["lead_hours"] = leads.astype(int)
    if f["origin_utc"].isna().any() or f["system"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Each forecast needs a valid origin and system name.")
    f["system"] = f["system"].astype(str).str.strip()
    f["forecast_temp_C"] = pd.to_numeric(f["forecast_temp_C"], errors="raise")

    if f.duplicated(["system", "origin_utc", "lead_hours"]).any():
        raise ValueError("Duplicate system/origin/lead forecast rows found.")

    # Keep an aware UTC valid time for auditability.
    f["valid_utc"] = f["origin_utc"] + pd.to_timedelta(f["lead_hours"], unit="h")
    # ECCC station CSVs are fixed LST.  Calgary LST = UTC-7 year-round.
    f["valid_lst"] = (
        f["valid_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
        + pd.to_timedelta(CALGARY_LST_UTC_OFFSET_HOURS, unit="h")
    )

    o = pd.read_csv(OBS)
    o["timestamp"] = pd.to_datetime(o["timestamp"], errors="raise")
    if o["timestamp"].duplicated().any():
        raise ValueError("Duplicate YYC observation timestamps found.")
    o = o[["timestamp", "temperature"]].rename(
        columns={"timestamp": "valid_lst", "temperature": "observed_temp_C"}
    )

    m = f.merge(o, on="valid_lst", how="left", validate="many_to_one")
    unmatched = int((~np.isfinite(m["observed_temp_C"])).sum())
    invalid_predictions = int((~np.isfinite(m["forecast_temp_C"])).sum())
    m = m[np.isfinite(m["forecast_temp_C"]) & np.isfinite(m["observed_temp_C"])].copy()
    if m.empty:
        raise ValueError("No forecasts matched finite YYC observations after UTC-to-LST conversion.")
    m["error_C"] = m["forecast_temp_C"] - m["observed_temp_C"]

    def _rmse(x: pd.Series) -> float:
        a = np.asarray(x, dtype=float)
        return float(np.sqrt(np.mean(a * a)))

    metrics = (
        m.groupby(["system", "lead_hours"], sort=True)
        .agg(
            n=("error_C", "size"),
            mae_C=("error_C", lambda x: float(np.mean(np.abs(x)))),
            rmse_C=("error_C", _rmse),
            bias_C=("error_C", "mean"),
        )
        .reset_index()
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "matched_forecasts.csv", index=False)
    metrics.to_csv(out / "eccc_metrics.csv", index=False)
    pd.DataFrame(
        [{
            "forecast_rows_input": len(f),
            "rows_without_finite_observation": unmatched,
            "rows_without_finite_forecast": invalid_predictions,
            "matched_finite_rows": len(m),
            "station_time_basis": "ECCC fixed LST (MST, UTC-7)",
        }]
    ).to_csv(out / "matching_audit.csv", index=False)

    print(metrics.to_string(index=False))
    print(f"\nMatched finite rows: {len(m)}; unmatched/non-finite observations: {unmatched}")


if __name__ == "__main__":
    main()
