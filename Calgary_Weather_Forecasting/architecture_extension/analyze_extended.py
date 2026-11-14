"""Rebuild the common-case scores, seasonal verification, station ablations and figures."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

if __package__:
    from .score_predictions import HORIZONS, RESULTS, SPATIAL, load_common_forecasts, metrics_by_lead, scores, seasonal_metrics
else:
    from score_predictions import HORIZONS, RESULTS, SPATIAL, load_common_forecasts, metrics_by_lead, scores, seasonal_metrics

FIG = Path(__file__).resolve().parents[1] / 'figures' / 'extended'


def station_ablations():
    design = pd.read_csv(SPATIAL / 'data/processed/model_ready_spatial.csv', parse_dates=['timestamp'])
    manifest = pd.read_csv(SPATIAL / 'data/processed/spatial_feature_manifest.csv')
    local = manifest.loc[manifest.feature_group == 'local', 'feature'].tolist()
    auxiliary = manifest.loc[manifest.feature_group == 'spatial']
    sets = {'Local only': local, 'Full spatial': manifest.feature.tolist(),
            'Spatial temperature only': local + auxiliary.loc[auxiliary.variable == 'temperature', 'feature'].tolist()}
    for station in auxiliary.source_station.unique():
        sets[f'Without {station}'] = local + auxiliary.loc[auxiliary.source_station != station, 'feature'].tolist()
    rows = []
    for h in HORIZONS:
        target = design[f'target_temp_{h}h']
        valid = design.timestamp + pd.Timedelta(hours=h)
        train = valid < '2024-01-01'
        val = (design.timestamp >= '2024-01-01') & (valid < '2025-01-01')
        test = (design.timestamp >= '2025-01-01') & (valid < '2026-01-01')
        raw = design[sets['Full spatial']]
        med = raw.loc[train].median()
        filled = raw.fillna(med)
        x = (filled - filled.loc[train].mean()) / filled.loc[train].std(ddof=0).replace(0., 1.)
        for name, columns in sets.items():
            model = LinearRegression().fit(x.loc[train, columns], target[train])
            for split, mask in [('validation', val), ('test', test)]:
                pred = model.predict(x.loc[mask, columns])
                rmse, mae, bias = scores(target[mask], pred)
                rows.append(dict(ablation=name, split=split, horizon_hours=h, n_features=len(columns),
                                 rmse_C=rmse, mae_C=mae, bias_C=bias))
    ablations = pd.DataFrame(rows)
    test = ablations.loc[ablations.split == 'test'].copy()
    full = test.loc[test.ablation == 'Full spatial'].set_index('horizon_hours').rmse_C
    test['full_rmse_C'] = test.horizon_hours.map(full)
    test['delta_rmse_vs_full_C'] = test.rmse_C - test.full_rmse_C
    return ablations, test


def plot_lines(data, group, y, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, series in data.groupby(group, sort=False):
        series = series.sort_values('horizon_hours')
        ax.plot(series.horizon_hours, series[y], marker='o', label=name)
    ax.set(xlabel='Forecast lead (h)', ylabel=ylabel, xticks=HORIZONS)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    common = load_common_forecasts()
    metrics = metrics_by_lead(common)
    seasonal = seasonal_metrics(common)
    ablations, deltas = station_ablations()
    for data, name in [(metrics, 'common_case_architecture_metrics'), (seasonal, 'seasonal_test_metrics'),
                       (ablations, 'spatial_ablation_ols'), (deltas, 'spatial_ablation_test_deltas')]:
        data.to_csv(RESULTS / f'{name}.csv', index=False)
    plot_lines(metrics.loc[~metrics.model.isin(['Local MLP', 'Local OLS'])], 'model', 'rmse_mean_C',
               'RMSE (°C)', 'architecture_rmse_common.png')
    plot_lines(seasonal.loc[seasonal.model == 'Spatial MLP'], 'season', 'rmse_C',
               'Spatial MLP RMSE (°C)', 'seasonal_spatial_mlp.png')
    matched = metrics.loc[metrics.model.isin(['Local MLP', 'Spatial MLP'])]
    plot_lines(matched, 'model', 'rmse_mean_C', 'RMSE (°C)', 'local_vs_spatial_mlp.png')
    paired = matched.pivot(index='horizon_hours', columns='model', values='rmse_mean_C')
    gain = 100 * (1 - paired['Spatial MLP'] / paired['Local MLP'])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(gain.index, gain, marker='o', color='#006e67')
    ax.axhline(0, color='0.4', linewidth=.8)
    ax.set(xlabel='Forecast lead (h)', ylabel='RMSE reduction from nearby stations (%)', xticks=HORIZONS)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(FIG / 'spatial_gain_percent.png', dpi=180)
    plt.close(fig)
    selected = deltas.loc[deltas.ablation.str.startswith('Without ') & deltas.horizon_hours.isin([6, 12, 24])].copy()
    selected['station'] = selected.ablation.str.removeprefix('Without ')
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    selected.pivot(index='station', columns='horizon_hours', values='delta_rmse_vs_full_C').plot.bar(ax=ax)
    ax.set(xlabel='', ylabel='RMSE change when station removed (°C)')
    ax.legend(title='Lead (h)')
    fig.tight_layout()
    fig.savefig(FIG / 'station_ablation.png', dpi=180)
    plt.close(fig)
    print(metrics.pivot(index='horizon_hours', columns='model', values='rmse_mean_C').round(3).to_string())


if __name__ == '__main__':
    main()
