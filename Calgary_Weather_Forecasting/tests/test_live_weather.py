import unittest
from datetime import datetime, timezone

from live_weather import HOUR, assemble, parse_reports


class LiveWeatherTests(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2026, 11, 10, 1, tzinfo=timezone.utc)
        self.rows = {self.t - lag * HOUR: {"temperature": float(lag)} for lag in range(24)}
        self.rows[self.t].update(dew_point=-5, wind_speed=0, relative_humidity=70, station_pressure=89)

    def test_hour_alignment_missingness_and_temperature_fallback(self):
        del self.rows[self.t - 4 * HOUR]
        reports = {"yyc": self.rows, "sundre": {self.t - 3 * HOUR: {"temperature": -13}}}
        with self.assertRaises(ValueError):
            assemble(reports, self.t, "spatial")
        seq = assemble(reports, self.t, "sequence-mlp")
        self.assertIsNone(seq.fields["lag_4__yyc__temperature"])
        snapshot = assemble(reports, self.t, "spatial", {self.t - 4 * HOUR: 7})
        self.assertEqual(snapshot.origin.strftime("%Y-%m-%d %H"), "2026-11-09 18")
        self.assertEqual(snapshot.fields["temp_lag_4"], 7)
        self.assertEqual(snapshot.metar_hours, 1)
        self.assertEqual(snapshot.fields["sundre_temp_lag_3"], -13)
        self.assertIsNone(snapshot.fields["sundre_temp_lag_0"])
        with self.assertRaises(ValueError):
            assemble(reports, self.t + 4 * HOUR, "sequence-tcn")

    def test_source_units_station_and_hour_filters(self):
        props = {"clim_id-value": "3031092", "obs_date_tm": self.t.isoformat(),
                 "stn_pres": 890, "stn_pres-uom": "hPa", "air_temp": 2, "air_temp-uom": "°C"}
        features = [{"properties": p} for p in (props, dict(props, **{"clim_id-value": "other", "air_temp": 9}),
                                                dict(props, obs_date_tm=self.t.replace(minute=10).isoformat()))]
        parsed = parse_reports(features, "3031092", self.t - HOUR, self.t + HOUR)
        self.assertEqual(parsed, {self.t: {"station_pressure": 89, "temperature": 2}})
