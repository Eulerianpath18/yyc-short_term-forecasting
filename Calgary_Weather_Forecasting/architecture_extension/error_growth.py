"""Compare 1–48 h error curves using the same least-squares objective for every form."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'spatial_extension' / 'data'
OUT = ROOT / 'architecture_extension' / 'results'
FIG = ROOT / 'figures' / 'extended'


def error_curve():
    design = pd.read_csv(DATA / 'processed/model_ready_spatial.csv', parse_dates=['timestamp'])
    features = pd.read_csv(DATA / 'processed/spatial_feature_manifest.csv').feature.tolist()
    obs = pd.read_csv(DATA / 'clean/yyc_hourly.csv', parse_dates=['timestamp']).set_index('timestamp').temperature
    rows = []
    for lead in range(1, 49):
        valid = design.timestamp + pd.Timedelta(hours=lead)
        target = valid.map(obs)
        train = (valid < '2024-01-01') & target.notna()
        test = (design.timestamp >= '2025-01-01') & (valid < '2026-01-01') & target.notna()
        medians = design.loc[train, features].median()
        x_train = design.loc[train, features].fillna(medians)
        means, stds = x_train.mean(), x_train.std(ddof=0).replace(0, 1)
        model = LinearRegression().fit((x_train - means) / stds, target[train])
        x_test = (design.loc[test, features].fillna(medians) - means) / stds
        forecasts = {'Persistence': design.loc[test, 'temp_lag_0'], 'Spatial OLS': model.predict(x_test)}
        for name, forecast in forecasts.items():
            rmse = np.sqrt(np.mean((target[test].to_numpy() - forecast) ** 2))
            rows.append(dict(model=name, horizon_hours=lead, n_samples=int(test.sum()), rmse_C=rmse))
    return pd.DataFrame(rows)


def fit_curves(curve):
    rows = []
    for model, group in curve.groupby('model', sort=False):
        h = group.horizon_hours.to_numpy(float)
        y = group.rmse_C.to_numpy(float)
        candidates = {}
        for label, x in [('a+b log(h)', np.log(h)), ('a+b sqrt(h)', np.sqrt(h))]:
            design = np.column_stack([np.ones_like(h), x])
            ab = np.linalg.lstsq(design, y, rcond=None)[0]
            candidates[label] = (ab, design @ ab)
        # Minimize squared residuals in degrees C, also for the power curve.
        def power(hours, a, b):
            return a * hours ** b
        ab, _ = curve_fit(power, h, y, p0=(1., .3), bounds=([0., -2.], [np.inf, 2.]))
        candidates['A h^alpha'] = (ab, power(h, *ab))
        for form, ((a, b), fitted) in candidates.items():
            sse = np.sum((y - fitted) ** 2)
            rows.append(dict(model=model, form=form, r2=1 - sse / np.sum((y - y.mean()) ** 2),
                             coefficient_a=a, coefficient_b=b, curve_rmse_C=np.sqrt(sse / len(h))))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    FIG.mkdir(exist_ok=True, parents=True)
    curve = error_curve()
    fits = fit_curves(curve)
    curve.to_csv(OUT / 'error_growth_1to48.csv', index=False)
    fits.to_csv(OUT / 'error_growth_fit_comparison.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.3, 4.7))
    for model, group in curve.groupby('model', sort=False):
        ax.plot(group.horizon_hours, group.rmse_C, label=model)
    ax.set(xlabel='Forecast lead (h)', ylabel='2025 RMSE (°C)')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'error_growth_1to48.png', dpi=180)
    plt.close(fig)
    print(fits.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
