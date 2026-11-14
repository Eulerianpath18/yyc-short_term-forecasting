"""Align published 2025 forecasts to station observations before scoring them."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / 'spatial_extension'
RESULTS = ROOT / 'architecture_extension' / 'results'
HORIZONS = (1, 6, 12, 24, 48)
SEEDS = (17, 2026, 2403)
KEY = ['timestamp', 'horizon_hours']


def scores(observed, forecast):
    error = np.asarray(forecast, dtype=float) - np.asarray(observed, dtype=float)
    if not error.size or not np.isfinite(error).all():
        raise ValueError('A score requires paired, finite observations and forecasts.')
    return np.sqrt(np.mean(error ** 2)), np.mean(np.abs(error)), np.mean(error)


def expected_keys(origins):
    return pd.MultiIndex.from_product([origins, HORIZONS], names=KEY).sort_values()


def verified_frame(frame, keys, observations, label):
    frame = frame.set_index(KEY, verify_integrity=True).sort_index()
    if not frame.index.equals(keys):
        raise ValueError(f'{label}: unexpected forecast origins or lead times.')
    origin = frame.index.get_level_values('timestamp')
    valid = origin + pd.to_timedelta(frame.index.get_level_values('horizon_hours'), unit='h')
    np.testing.assert_allclose(frame.observed_temp_C, observations.loc[valid], rtol=0, atol=2e-6,
                               err_msg=f'{label}: verifying temperatures differ from YYC.')
    if 'persistence_temp_C' in frame:
        np.testing.assert_allclose(frame.persistence_temp_C, observations.loc[origin], rtol=0, atol=1e-9)
    return frame


def load_common_forecasts():
    """Check full case coverage for every input, including sequence-only origins."""
    hourly = pd.read_csv(SPATIAL / 'data/clean/yyc_hourly.csv', parse_dates=['timestamp'])
    obs = hourly.set_index('timestamp', verify_integrity=True).temperature.sort_index()
    np.testing.assert_array_equal(obs.index, pd.date_range('2016-01-01', '2025-12-31 23:00', freq='h'))
    design = pd.read_csv(SPATIAL / 'data/processed/model_ready_spatial.csv', usecols=['timestamp'],
                         parse_dates=['timestamp'])
    end = pd.Timestamp('2026-01-01') - pd.Timedelta(hours=max(HORIZONS))
    origins = design.timestamp[design.timestamp.between('2025-01-01', end, inclusive='left')]
    keys = expected_keys(origins)
    eligible = obs.notna().copy()
    for lead in HORIZONS:
        eligible &= obs.shift(-lead).notna()
    seq_origins = obs.index[eligible & (obs.index >= '2025-01-01') & (obs.index < end)]
    seq_keys = expected_keys(seq_origins)
    if not keys.isin(seq_keys).all():
        raise ValueError('Engineered evaluation hours must be available in the sequence cohort.')

    def read(path):
        return pd.read_csv(path, parse_dates=['timestamp'])

    mlp = verified_frame(read(SPATIAL / 'results/spatial_mlp/final_test_predictions.csv'), keys, obs, 'MLPs')
    ols = verified_frame(read(SPATIAL / 'results/spatial_baselines/final_test_baseline_predictions.csv'), keys, obs, 'OLS')
    trees = verified_frame(read(RESULTS / 'lightgbm_test_predictions.csv'), keys, obs, 'LightGBM')
    common = pd.DataFrame({'observed': mlp.observed_temp_C, 'Persistence': mlp.persistence_temp_C,
                           'Local OLS': ols.local_ols_temp_C, 'Spatial OLS': ols.spatial_ols_temp_C,
                           'Local MLP': mlp.local_mlp_temp_C, 'Spatial MLP': mlp.spatial_mlp_temp_C,
                           'LightGBM': trees.lightgbm_temp_C})
    seq = read(RESULTS / 'sequence_test_predictions.csv')
    expected_runs = {(model, seed) for model in ('MLP', 'TCN') for seed in SEEDS}
    if set(zip(seq.model, seq.seed)) != expected_runs:
        raise ValueError('Unexpected sequence model/seed combinations.')
    for (model, seed), subset in seq.groupby(['model', 'seed']):
        label = f'Sequence {model} {seed}'
        full = verified_frame(subset, seq_keys, obs, label)
        if not np.isfinite(full.predicted_temp_C).all():
            raise ValueError(f'{label}: non-finite predictions.')
        common[label] = full.loc[keys, 'predicted_temp_C']
    return common


def metrics_by_lead(common):
    rows = []
    for h, group in common.groupby(level='horizon_hours'):
        models = {name: [name] for name in group if name != 'observed' and not name.startswith('Sequence ')}
        for kind in ('MLP', 'TCN'):
            name = f'Sequence {kind}'
            models[name] = [f'{name} {seed}' for seed in SEEDS]
        for model, columns in models.items():
            values = np.array([scores(group.observed, group[col]) for col in columns])
            rmse, mae, bias = values.mean(axis=0)
            rows.append({'model': model, 'horizon_hours': h, 'n_samples': len(group), 'rmse_mean_C': rmse,
                         'rmse_std_C': values[:, 0].std(ddof=1) if len(columns) > 1 else np.nan,
                         'mae_C': mae, 'bias_C': bias, 'seeds': len(columns)})
    return pd.DataFrame(rows)


def seasonal_metrics(common):
    seasons = np.array(['DJF', 'DJF', 'MAM', 'MAM', 'MAM', 'JJA', 'JJA', 'JJA', 'SON', 'SON', 'SON', 'DJF'])
    rows = []
    for h, group in common.groupby(level='horizon_hours'):
        valid = group.index.get_level_values('timestamp') + pd.Timedelta(hours=h)
        for season, subset in group.groupby(seasons[valid.month - 1]):
            for name in ('Persistence', 'Spatial OLS', 'Spatial MLP', 'LightGBM'):
                rmse, mae, bias = scores(subset.observed, subset[name])
                rows.append(dict(model=name, horizon_hours=h, season=season, n_samples=len(subset),
                                 rmse_C=rmse, mae_C=mae, bias_C=bias))
    return pd.DataFrame(rows)
