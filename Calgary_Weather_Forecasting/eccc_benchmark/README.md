# Scoring an external forecast file

This utility verifies supplied forecasts against YYC station observations. The Calgary project contains no archived operational forecasts or operational skill estimate.

Use one row per system, origin and lead. The temperature column uses degrees Celsius. This example shows the input layout with an illustrative value.

```csv
origin_utc,lead_hours,forecast_temp_C,system
2025-01-01T00:00:00Z,6,-10.2,example
```

```bash
python eccc_benchmark/benchmark_eccc.py forecasts.csv --output-dir eccc_benchmark/results
```

Valid time equals origin plus lead. YYC observations use fixed local standard time (UTC−7). The script joins on that clock year-round and preserves the UTC valid time. It reports RMSE, MAE and mean signed error for each system and lead on its available cases. Use identical origin/lead pairs across all systems, including the observation-based models, before comparing scores. The output audit counts missing observations and missing forecasts separately. See [the archive specification](archive_request.txt).
