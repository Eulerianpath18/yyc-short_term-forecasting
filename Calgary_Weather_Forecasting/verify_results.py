"""Verify the shipped 2025 forecasts, replay the two matched MLPs, and estimate paired RMSE gains.

Run from any directory with Python, NumPy and pandas. No network, GPU, or model fitting is required.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from architecture_extension.score_predictions import (
    HORIZONS, RESULTS, ROOT, SPATIAL, load_common_forecasts, metrics_by_lead, scores, seasonal_metrics,
)
from spatial_extension import preprocess_spatial as features
from spatial_extension.train_spatial import fit_preprocessor

DEST = ROOT / 'results' / 'verification'
MLP = SPATIAL / 'results' / 'spatial_mlp'


def check_table(actual, published, keys, tolerance=1e-5):
    expected = pd.read_csv(published).set_index(keys).sort_index()
    actual = actual.set_index(keys).sort_index()
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                  rtol=0, atol=tolerance, check_exact=False, obj=published.name)


def rebuild_design():
    """Reconstruct all inputs and labels from the six hourly series; retain the documented case rules."""
    airport = features.load_clean_station('yyc')
    original = pd.read_csv(ROOT / 'data/processed/eccc_calgary_hourly_clean.csv', parse_dates=['timestamp'])
    pd.testing.assert_frame_equal(airport, original)
    design, local_manifest = features.add_local_features(airport)
    spatial_manifest = features.add_spatial_features(design, airport)
    features.add_targets(design, airport)
    manifest = pd.DataFrame(local_manifest + spatial_manifest)
    pd.testing.assert_frame_equal(manifest, pd.read_csv(features.MANIFEST_PATH), check_dtype=False)
    local = [row['feature'] for row in local_manifest]
    all_features = manifest.feature.tolist()
    targets = [f'target_temp_{h}h' for h in HORIZONS]
    strict = design.dropna(subset=local + targets).set_index('timestamp')
    stored = pd.read_csv(features.OUTPUT_PATH, parse_dates=['timestamp']).set_index('timestamp')
    pd.testing.assert_frame_equal(strict[all_features + targets], stored, rtol=0, atol=1e-11)
    calm = airport.wind_speed.eq(0) & airport.wind_direction_10deg.isna()
    design.loc[calm, ['wind_dir_sin', 'wind_dir_cos']] = 0.
    expanded = design.dropna(subset=local + targets).set_index('timestamp')
    local_file = pd.read_csv(ROOT / 'data/processed/model_ready.csv', parse_dates=['timestamp']).set_index('timestamp')
    pd.testing.assert_frame_equal(expanded[local + targets], local_file, rtol=0, atol=1e-11)
    return strict, expanded, local, all_features


def replay_matched(common):
    strict, expanded, local_features, all_features = rebuild_design()
    saved_stats = pd.read_csv(MLP / 'preprocessing_stats.csv').set_index(['horizon_hours', 'model', 'feature'])
    rows = []
    with np.load(MLP / 'checkpoints/matched_weights.npz', allow_pickle=False) as weights:
        for h in HORIZONS:
            target = f'target_temp_{h}h'
            train = strict.loc[strict.index + pd.Timedelta(hours=h) < '2024-01-01']
            valid = expanded.index + pd.Timedelta(hours=h)
            test = expanded.loc[(expanded.index >= '2025-01-01') & (valid < '2026-01-01')]
            matched = common.xs(h, level='horizon_hours')
            primary = test.index.isin(strict.index)
            pd.testing.assert_index_equal(test.index[primary], matched.index)
            med, mean, std = fit_preprocessor(train[all_features])
            scaled = (test[all_features].fillna(med) - mean) / std
            target_mean = train[target].mean()
            target_std = train[target].std(ddof=0)
            preds = {}
            for kind, columns in [('local', local_features), ('spatial', all_features)]:
                model = f'{kind.title()} MLP'
                expected = saved_stats.loc[(h, model), :].reindex(columns)
                np.testing.assert_allclose(expected[['train_median', 'train_mean_after_imputation',
                                                     'train_std_after_imputation']],
                                           np.stack([s[columns] for s in (med, mean, std)], axis=1),
                                           rtol=0, atol=1e-9)
                prefix = f'{kind}_{h}_network.'
                x = scaled[columns].to_numpy(dtype=np.float32)
                hidden = np.maximum(x @ weights[prefix + '0.weight'].T + weights[prefix + '0.bias'], 0)
                y = (hidden @ weights[prefix + '2.weight'].T + weights[prefix + '2.bias']).ravel()
                preds[kind] = y * float(target_std) + float(target_mean)
                # CSV rounding and CPU float32 accumulation order can vary slightly.
                np.testing.assert_allclose(preds[kind][primary], matched[model], rtol=0, atol=3e-5,
                                           err_msg=f'{model} {h} h checkpoint replay')
            for cohort, mask in [('Matched', primary), ('Calm additions', ~primary),
                                 ('Including calm', np.ones(len(test), dtype=bool))]:
                y = test.loc[mask, target]
                lrmse = scores(y, preds['local'][mask])[0]
                srmse = scores(y, preds['spatial'][mask])[0]
                rows.append(dict(horizon_hours=h, cohort=cohort, n_samples=len(y), local_rmse_C=lrmse,
                                 spatial_rmse_C=srmse, gain_percent=100 * (1 - srmse / lrmse)))
    return pd.DataFrame(rows)


def paired_rmse_intervals(common, reference, candidate, n_draws=5000, seed=2403, block_lengths=(3, 7, 14)):
    """Paired circular moving blocks of calendar days, retaining missing hours and covariance across leads."""
    days = pd.date_range('2025-01-01', '2025-12-31', freq='D')
    daily = []
    for h in HORIZONS:
        group = common.xs(h, level='horizon_hours')
        losses = group[[reference, candidate]].sub(group.observed, axis=0).pow(2)
        losses['n'] = 1
        sums = losses.groupby(losses.index.normalize()).sum().reindex(days, fill_value=0.)
        daily.append(sums.to_numpy())
    daily = np.stack(daily, axis=1)  # day, lead, (two squared-error sums, count)
    rows = []
    for block in block_lengths:
        rng = np.random.default_rng(seed)
        starts = rng.integers(len(days), size=(n_draws, int(np.ceil(len(days) / block))))
        # Sum complete blocks directly from the calendar grid. Truncate the last one to 365 days.
        def totals(length):
            return sum(np.roll(daily, -offset, axis=0) for offset in range(length))
        sample = totals(block)[starts[:, :-1]].sum(axis=1)
        last_length = len(days) - (starts.shape[1] - 1) * block
        sample += totals(last_length)[starts[:, -1]]
        rmse = np.sqrt(sample[:, :, :2] / sample[:, :, 2:])
        differences = rmse[:, :, 0] - rmse[:, :, 1]
        percentages = 100 * (1 - rmse[:, :, 1] / rmse[:, :, 0])
        actual = np.sqrt(daily[:, :, :2].sum(axis=0) / daily[:, :, 2:].sum(axis=0))
        diff_ci = np.quantile(differences, [.025, .975], axis=0)
        pct_ci = np.quantile(percentages, [.025, .975], axis=0)
        for j, h in enumerate(HORIZONS):
            local, spatial = actual[j]
            rows.append(dict(horizon_hours=h, block_days=block, draws=n_draws, seed=seed,
                             n_samples=int(daily[:, j, 2].sum()), reference_rmse_C=local, candidate_rmse_C=spatial,
                             gain_C=local - spatial, lower_C=diff_ci[0, j], upper_C=diff_ci[1, j],
                             gain_percent=100 * (1 - spatial / local), lower_percent=pct_ci[0, j],
                             upper_percent=pct_ci[1, j]))
    return pd.DataFrame(rows)


def bootstrap_gains(common, n_draws=5000, seed=2403):
    intervals = paired_rmse_intervals(common, 'Local MLP', 'Spatial MLP', n_draws, seed)
    return intervals.rename(columns={'reference_rmse_C': 'local_rmse_C', 'candidate_rmse_C': 'spatial_rmse_C'})


def main():
    common = load_common_forecasts()
    check_table(metrics_by_lead(common), RESULTS / 'common_case_architecture_metrics.csv', ['model', 'horizon_hours'])
    check_table(seasonal_metrics(common), RESULTS / 'seasonal_test_metrics.csv', ['model', 'horizon_hours', 'season'])
    sensitivity = replay_matched(common)
    gains = bootstrap_gains(common)
    weekly = gains.loc[gains.block_days == 7].set_index('horizon_hours')
    published = pd.read_csv(MLP / 'final_test_spatial_gain.csv').set_index('horizon_hours')
    np.testing.assert_allclose(weekly[['n_samples', 'gain_C', 'gain_percent']],
                               published[['n_samples', 'rmse_reduction_C', 'rmse_reduction_percent']], rtol=0, atol=1e-5)
    DEST.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(DEST / 'calm_wind_sensitivity.csv', index=False)
    gains.to_csv(DEST / 'paired_rmse_gains.csv', index=False)
    print('Verified all six hourly grids, both engineered datasets, forecast keys and labels,')
    print('40 common-case model/lead scores, seasonal scores, and 10 matched MLP weight sets.')
    print(weekly[['n_samples', 'local_rmse_C', 'spatial_rmse_C', 'gain_percent', 'lower_percent',
                  'upper_percent']].round(2).to_string())


if __name__ == '__main__':
    main()
