"""Compare manual-vector forecasts to the original timestamped predictions."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from architecture_extension.sequence_data import build
from predict import HORIZONS, SEEDS, _json_input, feature_schema, predict_temperature

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / "spatial_extension"


def read_index(path):
    return pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp")


class ManualForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spatial = json.loads((ROOT / "examples/spatial.json").read_text())
        cls.local = json.loads((ROOT / "examples/local.json").read_text())
        cls.sequence = json.loads((ROOT / "examples/sequence.json").read_text())

    def test_engineered_replay_all_leads_and_missing_auxiliary(self):
        matched = read_index(SPATIAL / "data/processed/model_ready_spatial.csv").loc["2025"]
        forecasts = read_index(SPATIAL / "results/spatial_mlp/final_test_predictions.csv")
        for model in ("local", "spatial", "standalone-local"):
            names = [name for name, _, _ in feature_schema(model)]
            if model == "standalone-local":
                design = read_index(ROOT / "data/processed/model_ready.csv").loc["2025"]
                archive = read_index(ROOT / "results/mlp/final_test_predictions.csv")
                column = "mlp_pred_C"
                extra = design.index.difference(matched.index)[0]  # A calm hour omitted by the matched cohort.
            else:
                design, archive = matched, forecasts
                column = model + "_mlp_temp_C"
                extra = design.index[design.bow_valley_temp_lag_0.isna()][0]

            for origin in (design.index[0], design.index[2500], design.index[-1], extra):
                ordered = design.loc[origin, names].tolist()
                named = dict(zip(reversed(names), reversed(ordered)))
                expected = archive.loc[origin].set_index("horizon_hours").loc[list(HORIZONS), column]
                for vector in (ordered, named):
                    with self.subTest(model=model, origin=origin, vector_type=type(vector).__name__):
                        actual = predict_temperature(vector, model=model)
                        np.testing.assert_allclose(list(actual.values()), expected.to_numpy(), atol=3e-5, rtol=0)

    def test_sequence_checkpoints_and_mask_order(self):
        source = build(False)
        archive = read_index(ROOT / "architecture_extension/results/sequence_test_predictions.csv")
        positions = np.flatnonzero(source.origin_timestamps.year == 2025)

        def history(index):
            origin = source.origins[index]
            return source.raw[origin - 23:origin + 1]

        missing = next(i for i in positions if np.isnan(history(i)).any())
        for i in (positions[0], positions[-1], missing):
            timestamp = source.origin_timestamps[i]
            vector = np.concatenate([history(i).ravel(), source.static[i]])
            observed_runs = archive.loc[timestamp]
            for model in ("MLP", "TCN"):
                for seed in SEEDS:
                    with self.subTest(origin=timestamp, model=model, seed=seed):
                        run = observed_runs.loc[(observed_runs.model == model) & (observed_runs.seed == seed)]
                        expected = run.set_index("horizon_hours").loc[list(HORIZONS), "predicted_temp_C"]
                        actual = predict_temperature(vector, model="sequence-" + model.lower(), seed=seed)
                        np.testing.assert_allclose(list(actual.values()), expected, atol=4e-5, rtol=0)
        structured = predict_temperature(self.sequence, model="sequence-tcn")
        flat = np.concatenate([np.asarray(self.sequence["history"]).ravel(), self.sequence["calendar"]])
        self.assertEqual(structured, predict_temperature(flat, model="sequence-tcn"))

    def test_user_edits_and_validation(self):
        original = predict_temperature(self.spatial, horizons=6)[6]
        changed = dict(self.spatial, bow_valley_temp_lag_0=-10.5)
        self.assertAlmostEqual(original, -14.0278, places=4)
        self.assertAlmostEqual(predict_temperature(changed, horizons=[6])[6], -13.9883, places=4)
        self.assertTrue(np.isfinite(predict_temperature(dict(self.spatial, bow_valley_temp_lag_0=None), horizons=6)[6]))
        invalid = [[1, 2], dict(self.spatial, temp_lag_0=None), dict(self.spatial, station_pressure=np.inf),
                   dict(self.spatial, dew_point=True), dict(self.spatial, target_temp_6h=5)]
        for vector in invalid:
            with self.assertRaises(ValueError):
                predict_temperature(vector)
        for options in (dict(horizons=3), dict(horizons=6.5), dict(seed=17)):
            with self.assertRaises(ValueError):
                predict_temperature(self.local, model="local", **options)
        with self.assertRaises(ValueError):
            predict_temperature(self.sequence, model="sequence-tcn", seed=17.0)
        for text in ('{"a":1,"a":2}', '[NaN]'):
            with self.assertRaises(ValueError):
                _json_input(text)
        lengths = [len(feature_schema(m)) for m in ("local", "spatial", "sequence-mlp")]
        self.assertEqual(lengths, [34, 67, 580])

    def test_cli_file_literal_stdin_and_invalid_vector(self):
        args = [sys.executable, str(ROOT / "predict.py"), "--model", "spatial", "--lead", "6", "--json"]
        cases = [
            (["--input", str(ROOT / "examples/spatial.json")], None),
            (["--features", json.dumps(list(self.spatial.values()))], None),
            ([], json.dumps(self.spatial)),
        ]
        with tempfile.TemporaryDirectory() as cwd:
            for extra, stdin in cases:
                run = subprocess.run(args + extra, input=stdin, text=True, capture_output=True, check=True, cwd=cwd)
                forecast = json.loads(run.stdout)["temperature_C_by_lead_h"]["6"]
                self.assertAlmostEqual(forecast, -14.0278, places=4)
            fail = subprocess.run(args + ["--features", "[1]"], text=True, capture_output=True, cwd=cwd)
            self.assertEqual(fail.returncode, 2)
            self.assertIn("67", fail.stderr)


if __name__ == "__main__":
    unittest.main()
