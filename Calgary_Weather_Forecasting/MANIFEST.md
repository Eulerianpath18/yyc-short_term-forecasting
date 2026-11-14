# File guide

| Location | Contents |
| --- | --- |
| [README.md](README.md) | Main finding, installation, commands and case definitions |
| [Report](report/Calgary_Temperature_Forecasting.pdf) and [editable source](report/Calgary_Temperature_Forecasting.tex) | Technical study with station geometry, matched verification, error diagnostics, case weeks, error-growth curves and a worked manual forecast |
| [live_weather.py](live_weather.py) | Imports time-aligned YYC and regional observations from ECCC for live forecasts |
| [forecast_gui.py](forecast_gui.py) | Native desktop forecast window with editable station observations, lead selection and predicted YYC temperatures |
| [predict.py](predict.py), [weights](inference/forecast_weights.npz) and [examples](examples) | Manual vector inference for all trained neural models, with NumPy |
| [export_prediction_weights.py](export_prediction_weights.py) and [regression checks](tests/test_predict.py) | Refresh portable weights and compare their outputs against archived forecasts |
| [Report diagnostics](report/analyze_diagnostics.py) and [tables](report/results) | Reproduce the supplemental benchmarks, paired error distributions and case studies |
| [verify_results.py](verify_results.py) | Offline verification, matched checkpoint replay, bootstrap and calm-wind sensitivity |
| [data/processed](data/processed) and [results](results) | Standalone airport dataset, baselines, MLP results and verification outputs |
| [spatial_extension](spatial_extension) | Station inventory, six cleaned hourly datasets, feature manifest, scripts, matched OLS and MLP outputs |
| [architecture_extension](architecture_extension) | PCR, LightGBM, sequence MLP and TCN scripts, common-case scoring, seasonal verification, station ablations and lead-time curve fits |
| [figures](figures) | Standalone and regional experiment plots |
| [eccc_benchmark](eccc_benchmark) | Input specification and verification utility for externally supplied forecasts |

The root scripts `download_and_preprocess_weather.py`, `models.py`, `train.py`, `train_mlp.py` and `evaluate.py` implement the original airport-only experiment. Regional and sequence results use the cohorts identified in the README.
