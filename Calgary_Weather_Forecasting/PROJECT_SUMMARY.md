# Calgary temperature forecasting

## Résumé wording

Built a Python/PyTorch temperature forecasting pipeline using ten years of hourly observations from six Alberta weather stations. Adding five regional stations reduced six-hour Calgary airport forecast RMSE by 13.0% versus a matched airport-only neural network (2.68°C to 2.33°C across 8,102 held-out 2025 forecasts).

## Results behind the claim

- Both MLPs use one 32-unit hidden layer and a common training and validation protocol. They were trained separately with 34 or 67 predictors. At 6 hours, the matched persistence, airport OLS and six-station OLS benchmarks had RMSEs of 5.286°C, 3.156°C and 2.844°C. The matched network errors were 2.676°C and 2.328°C.
- The six-station MLP reduced test RMSE at the four evaluated leads from 1 through 24 hours. At 48 hours its RMSE was 5.782°C, compared with 5.750°C for the airport model. The seven-day paired block interval for the 48-hour relative gain crosses zero.
- LightGBM had the lowest observed 24- and 48-hour RMSE, with a 24-hour margin of about 0.002°C over the six-station MLP. The compact TCN used a 15-epoch budget and roughly 3% as many parameters as the flattened sequence MLP. Both score higher than the engineered MLP at every lead in this study.
- Bow Valley made the largest leave-one-station-out OLS contribution at 6, 12 and 24 hours. At 48 hours the spatial MLP's winter RMSE was 8.53°C, compared with 3.82°C in summer, by verifying date in 2025.
- Direct spatial OLS forecast error grew sublinearly over 1–48 hours. A power curve fitted on the RMSE scale had the highest in-sample R² (0.9927), followed by a logarithmic curve (0.9873) and a square-root curve (0.9793). The evidence covers one location and test year.

At six hours the regional model also reduced the count of absolute errors of at least 5°C from 496 to 326. The expanded [diagnostics](report/results) preserve the underlying error frequencies and forecast case counts. The [desktop forecast demo](README.md#open-the-desktop-forecast-window) applies the saved weights to new input and can be demonstrated without retraining.

The comparison is between observation-based forecasts. ECCC supplied the station data. No operational weather-model forecast was scored. See the [technical report](report/Calgary_Temperature_Forecasting.pdf) for methods, model results and limitations.
