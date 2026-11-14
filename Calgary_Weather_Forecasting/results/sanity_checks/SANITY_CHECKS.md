# Forecast verification

## Checks performed on this release

- Rebuilt both supervised datasets from the six cleaned station grids. Each file contains 87,672 hourly timestamp slots with no gaps in the time index or duplicate timestamps. Missing measurements remain explicit. Local features and all five verifying targets match the bundled data, as do the auxiliary features and their missing values.
- Verified unique origin/lead keys, complete expected test coverage and observed temperature at each valid time for the matched MLP, OLS, LightGBM and six sequence runs. Persistence was checked against the airport temperature at the origin. Recomputed common-case RMSE, MAE, signed bias, between-seed dispersion and seasonal scores from row-level predictions.
- Replayed all 21 supplied neural-network weight sets using CPU NumPy operations and compared forecasts with the saved predictions. The ten matched MLPs agreed within 0.00003°C across their 8,102 primary test origins per lead. The five standalone airport networks and six sequence checkpoints agreed within 0.00002°C. Numerical agreement concerns inference from fitted weights.
- Refit matched OLS, PCR, station ablations and all 48 lead-specific spatial OLS regressions from the included training data. The three error-growth candidates now share a least-squares objective in degrees Celsius. This changes the ranking of the fitted power and logarithmic curves.
- Scored the same matched MLP checkpoints on the additional calm-wind cases, using training-period scalers and a zero direction vector. Estimated within-year paired uncertainty with 5,000 circular moving-block draws per block length (3, 7 and 14 days).
- Checked the external forecast scorer against winter and summer timestamp fixtures. A 01 UTC origin plus a six-hour lead matched 00 fixed local standard time in both cases. A fractional lead of 6.5 hours is rejected. These fixtures test the time join only.

`python verify_results.py` runs the data reconstruction, common-case and seasonal score checks, ten matched checkpoint replays, sensitivity calculation and block estimates. The result tables are in `results/verification/`. Native checkpoint replay commands for the other neural models, regression refits and the LightGBM training command are in the root README. The neural training histories and LightGBM predictions came from the original fitted experiments. The review used the supplied station snapshot.

## Earlier experiment records

The CSVs and text files in this directory also preserve the original grid, sequence-alignment, checkpoint RMSE and shuffled-target checks. The recorded six-hour shuffled-target OLS control increased validation RMSE from 2.70°C to 11.94°C. The earlier `local_mlp_checkpoint` entries in `reproduction_check.csv` use the standalone airport model. Use `spatial_extension/results/spatial_mlp/` for the airport baseline in the matched regional comparison.
