"""Recompute the report's supplementary 2025 diagnostics from the bundled data.

From the project root, run: python -m report.analyze_diagnostics
"""
from pathlib import Path

import numpy as np
import pandas as pd

from architecture_extension.score_predictions import SPATIAL, load_common_forecasts, scores
from verify_results import paired_rmse_intervals

OUT = Path(__file__).resolve().parent / "results"
LOCAL, REGIONAL = "Local MLP", "Spatial MLP"
SEASON_BY_MONTH = np.array(["DJF", "DJF", "MAM", "MAM", "MAM", "JJA", "JJA", "JJA", "SON", "SON", "SON", "DJF"])


def reference_forecasts(common, temperature):
    training = temperature.loc[temperature.index < "2024-01-01"].dropna()
    calendar = training.groupby([training.index.month, training.index.hour]).agg(["mean", "count"])
    calendar.index.names = ["month", "hour_lst"]
    if len(calendar) != 288:
        raise ValueError("Missing month/hour cell in the training climate reference.")

    def normal(times):
        keys = pd.MultiIndex.from_arrays([times.month, times.hour], names=calendar.index.names)
        return calendar["mean"].loc[keys].to_numpy()

    rows = []
    for horizon, group in common.groupby(level="horizon_hours"):
        origins = group.index.get_level_values("timestamp")
        expected = normal(origins + pd.Timedelta(hours=horizon))
        predictions = {
            "Month-hour mean": expected,
            "Anomaly persistence": group.Persistence.to_numpy() + expected - normal(origins),
        }
        for name, prediction in predictions.items():
            rmse, mae, bias = scores(group.observed, prediction)
            rows.append(dict(horizon_hours=horizon, model=name, n=len(group), rmse_C=rmse, mae_C=mae, bias_C=bias))
    return calendar.reset_index(), pd.DataFrame(rows)


def pair_stats(group):
    observed = group.observed.to_numpy()
    local = group[LOCAL].to_numpy() - observed
    regional = group[REGIONAL].to_numpy() - observed
    local_rmse, regional_rmse = np.sqrt(np.mean(local**2)), np.sqrt(np.mean(regional**2))
    return dict(
        n=len(group), local_rmse_C=local_rmse, spatial_rmse_C=regional_rmse,
        gain_C=local_rmse - regional_rmse, gain_pct=100 * (1 - regional_rmse / local_rmse),
        local_mae_C=np.mean(abs(local)), spatial_mae_C=np.mean(abs(regional)),
        local_bias_C=local.mean(), spatial_bias_C=regional.mean(),
        closer_pct=100 * np.mean(abs(regional) < abs(local)),
        local_sse=np.sum(local**2), spatial_sse=np.sum(regional**2),
    )


def distribution_diagnostics(common, design):
    train = design.loc[design.index + pd.Timedelta(hours=6) < "2024-01-01"]
    contrast = (train.bow_valley_temp_lag_0 - train.temp_lag_0).abs().dropna()
    quantiles = [0.25, 0.50, 0.75]
    # Temperature is measured in tenths of a degree. Round before grouping decimal ties.
    cuts = contrast.quantile(quantiles).round(1).to_numpy()
    seasons, regimes, tails, gradients, totals = [], [], [], [], []

    for horizon, group in common.groupby(level="horizon_hours"):
        group = group.droplevel("horizon_hours")
        valid_times = group.index + pd.Timedelta(hours=horizon)
        totals.append(dict(horizon_hours=horizon, **pair_stats(group)))
        for season, part in group.groupby(SEASON_BY_MONTH[valid_times.month - 1]):
            seasons.append(dict(horizon_hours=horizon, season=season, **pair_stats(part)))
        bins = pd.cut(group.observed, [-np.inf, -10, 0, 10, 20, np.inf], right=False,
                      labels=["below -10", "-10 to 0", "0 to 10", "10 to 20", "20 or above"])
        for band, part in group.groupby(bins, observed=True):
            regimes.append(dict(horizon_hours=horizon, valid_temperature_band=band, **pair_stats(part)))

        for model in (LOCAL, REGIONAL):
            errors = (group[model] - group.observed).abs()
            misses = errors.ge(5)
            tails.append(dict(
                horizon_hours=horizon, model=model, n=len(errors), median_abs_C=errors.median(),
                q90_abs_C=errors.quantile(.90), q95_abs_C=errors.quantile(.95),
                n_at_least_5_C=int(misses.sum()), pct_at_least_5_C=100 * misses.mean(),
            ))

        if horizon == 6:
            current = design.loc[group.index]
            spread = (current.bow_valley_temp_lag_0 - current.temp_lag_0).abs().round(1)
            quartile = pd.cut(spread, [-np.inf, *cuts, np.inf], labels=["Q1", "Q2", "Q3", "Q4"])
            for band, part in group.groupby(quartile, observed=True):
                gradients.append(dict(band=band, **pair_stats(part)))
            if spread.isna().any():
                gradients.append(dict(band="Missing", **pair_stats(group.loc[spread.isna()])))

    return {
        "paired_errors": pd.DataFrame(totals), "seasonal_gains": pd.DataFrame(seasons),
        "temperature_bands": pd.DataFrame(regimes), "error_tails": pd.DataFrame(tails),
        "bow_valley_contrast": pd.DataFrame(gradients),
        "contrast_thresholds": pd.DataFrame({"quantile": quantiles, "contrast_C": cuts, "n_train": len(contrast)}),
    }


def example_weeks(common, temperature):
    forecasts = common.xs(6, level="horizon_hours")[["observed", LOCAL, REGIONAL]].copy()
    forecasts.index += pd.Timedelta(hours=6)
    rows, metrics = [], []
    for date in ("2025-02-01", "2025-07-01"):
        hours = pd.date_range(date, periods=168, freq="h", name="valid_lst")
        week = forecasts.reindex(hours)
        metrics.append(dict(week_start=date, **pair_stats(week.dropna())))
        week["observed"] = temperature.loc[hours]
        week["week_start"] = date
        week["hours_since_start"] = np.arange(len(hours))
        rows.append(week.reset_index())
    return pd.concat(rows, ignore_index=True), pd.DataFrame(metrics)


def main():
    common = load_common_forecasts()
    airport = pd.read_csv(SPATIAL / "data/clean/yyc_hourly.csv", parse_dates=["timestamp"])
    temperature = airport.set_index("timestamp").temperature
    design = pd.read_csv(SPATIAL / "data/processed/model_ready_spatial.csv", parse_dates=["timestamp"],
                         usecols=["timestamp", "bow_valley_temp_lag_0", "temp_lag_0"]).set_index("timestamp")
    tables = distribution_diagnostics(common, design)
    tables["month_hour_training"], tables["calendar_benchmarks"] = reference_forecasts(common, temperature)
    tables["example_weeks"], tables["example_week_scores"] = example_weeks(common, temperature)
    tables["tree_vs_spatial_intervals"] = paired_rmse_intervals(common, REGIONAL, "LightGBM", block_lengths=(7,))
    OUT.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    print(tables["paired_errors"][["horizon_hours", "n", "gain_pct", "closer_pct"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
