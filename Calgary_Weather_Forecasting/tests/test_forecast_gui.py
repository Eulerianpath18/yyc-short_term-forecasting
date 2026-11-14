"""Check desktop entry, encoded inputs, lead selection and saved-weight inference."""
import json
from pathlib import Path
import unittest

try:
    import tkinter as tk
    from forecast_gui import ForecastWindow, MODELS, origin_and_calendar, wind_components
except ImportError:
    tk = None

from predict import HORIZONS, SEEDS, predict_temperature

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(tk is None, "Tkinter is unavailable")
class PhysicalInputs(unittest.TestCase):
    def test_calendar_and_wind_reproduce_the_engineered_example(self):
        sample = json.loads((ROOT / "examples/spatial.json").read_text())
        _, calendar = origin_and_calendar("2025-01-01", "00")
        for key, value in calendar.items():
            self.assertAlmostEqual(value, sample[key])
        for value, key in zip(wind_components("40", 13), ("wind_dir_sin", "wind_dir_cos")):
            self.assertAlmostEqual(value, sample[key])
        self.assertEqual(wind_components("", 0), (0., 0.))
        with self.assertRaises(ValueError):
            wind_components("", 5)
        with self.assertRaises(ValueError):
            origin_and_calendar("2025-02-29", "00")


@unittest.skipIf(tk is None, "Tkinter is unavailable")
class DesktopForecasts(unittest.TestCase):
    def setUp(self):
        try:
            self.window = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Graphical display unavailable: {exc}")
        self.addCleanup(self.window.destroy)
        self.ui = ForecastWindow(self.window)
        self.window.update()

    def change_model(self, label):
        self.ui.model_box.set(label)
        self.ui.model_box.event_generate("<<ComboboxSelected>>")
        self.window.update()

    def test_gui_uses_every_saved_checkpoint(self):
        for label, model in MODELS.items():
            self.change_model(label)
            sequence = model.startswith("sequence-")
            example = "sequence" if sequence else ("spatial" if model == "spatial" else "local")
            sample = json.loads((ROOT / f"examples/{example}.json").read_text())
            for seed in (SEEDS if sequence else [None]):
                if sequence:
                    self.ui.seed.set(str(seed))
                self.ui.run_button.invoke()
                expected = predict_temperature(sample, model=model, seed=seed)
                with self.subTest(model=model, seed=seed):
                    for h in HORIZONS:
                        self.assertEqual(self.ui.answers[h][1].get(), f"{expected[h]:.3f}")

    def test_edits_leads_errors_and_reload(self):
        ui = self.ui
        for h in HORIZONS:
            if h != 6:
                ui.lead_buttons[h].invoke()
        ui.fields["bow_valley_temp_lag_0"].set("-10.5")
        self.assertEqual(ui.answers[6][1].get(), "—")
        ui.run_button.invoke()
        self.assertEqual(ui.answers[6][1].get(), "-13.988")
        self.assertEqual(ui.answers[6][0].get(), "01 Jan 06:00")
        self.assertEqual(ui.answers[48][1].get(), "—")
        ui.fields["bow_valley_temp_lag_0"].set("")
        ui.run_button.invoke()
        self.assertIn(6, ui.profile)
        ui.fields["relative_humidity"].set("120")
        ui.run_button.invoke()
        self.assertIn("100", ui.status.get())
        self.assertFalse(ui.profile)
        ui.load_example()
        ui.run_button.invoke()
        self.assertEqual(ui.answers[6][1].get(), "-14.028")
        ui.lead_buttons[6].invoke()
        ui.run_button.invoke()
        self.assertIn("at least one", ui.status.get())
        ui.clear()
        self.assertEqual(ui.fields["temp_lag_0"].get(), "")
        self.assertEqual(ui.day.get(), "")

    def test_station_switch_preserves_values_and_updates_physical_time(self):
        ui = self.ui
        ui.tabs.select(2)
        page = ui.canvases[ui.tabs.select()]
        grid = next(iter(page.winfo_children()))
        menu = next(w for w in grid.winfo_children() if w.winfo_class() == "TCombobox")
        menu.set("Sundre A")
        menu.event_generate("<<ComboboxSelected>>")
        self.window.update()
        ui.fields["sundre_temp_lag_0"].set("-20")
        menu.set("Bow Valley")
        menu.event_generate("<<ComboboxSelected>>")
        self.assertEqual(ui.fields["sundre_temp_lag_0"].get(), "-20")
        ui.day.set("2024-12-31")
        ui.hour.set("20")
        ui.run_button.invoke()
        self.assertEqual(ui.answers[6][0].get(), "01 Jan 02:00")


if __name__ == "__main__":
    unittest.main()
