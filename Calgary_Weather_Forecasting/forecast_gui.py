"""Open the offline Calgary Airport forecast desktop application."""
from datetime import datetime, timedelta
import json
import math
from queue import Empty, Queue
from threading import Thread
from live_weather import load_latest, MST
from pathlib import Path
import tkinter as tk
from tkinter import font, messagebox, ttk

try:
    from predict import CALENDAR, HORIZONS, SEEDS, feature_schema, predict_temperature
except ModuleNotFoundError as exc:
    if __name__ != "__main__" or exc.name != "numpy":
        raise
    import sys
    window = tk.Tk()
    window.withdraw()
    messagebox.showerror("Install NumPy", "Install NumPy for this Python interpreter, then reopen the forecast window.\n\n"
                         + f'"{sys.executable}" -m pip install numpy', parent=window)
    window.destroy()
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent
MODELS = {
    "Six-station MLP": "spatial", "Airport MLP (matched)": "local",
    "Airport MLP (standalone)": "standalone-local", "Sequence MLP": "sequence-mlp",
    "Sequence TCN": "sequence-tcn",
}
STATIONS = {
    "Calgary Intl. Airport": "yyc", "Bow Valley": "bow_valley", "Sundre A": "sundre",
    "Drumheller East": "drumheller", "Claresholm": "claresholm", "Lacombe CDA 2": "lacombe",
}
WEATHER = {
    "dew_point": "Dew point (°C)", "relative_humidity": "Relative humidity (%)",
    "station_pressure": "Station pressure (kPa)", "wind_speed": "Wind speed (km/h)",
}
BG, WHITE, INK, SOFT = "#eff3f5", "#ffffff", "#173544", "#4d646d"
TEAL, NAVY, LINE, RED = "#006c63", "#143844", "#cbd9da", "#963d26"


def origin_and_calendar(day, hour):
    try:
        origin = datetime.strptime(f"{day.strip()} {hour.strip()}", "%Y-%m-%d %H")
    except ValueError as exc:
        raise ValueError("Enter the issue date as YYYY-MM-DD and the local standard hour from 00 to 23.") from exc
    daily = math.tau * origin.hour / 24
    annual = math.tau * origin.timetuple().tm_yday / 365.25
    return origin, dict(zip(CALENDAR, (math.sin(daily), math.cos(daily), math.sin(annual), math.cos(annual))))


def reading(text, label, optional=False):
    if optional and not text.strip():
        return None
    try:
        value = float(text)
        if math.isfinite(value):
            return value
    except ValueError:
        pass
    raise ValueError(f"Enter a finite number for {label}.")


def wind_components(text, speed):
    if not text.strip() and speed == 0:
        return 0.0, 0.0
    direction = reading(text, "airport wind direction in degrees (leave blank for calm wind)")
    if not 0 <= direction <= 360:
        raise ValueError("Airport wind direction must be between 0 and 360 degrees.")
    angle = math.radians(direction)
    return math.sin(angle), math.cos(angle)


class ForecastWindow:
    def __init__(self, root):
        self.root = root
        root.title("Calgary Airport | Temperature forecasts")
        width = min(1060, root.winfo_screenwidth() - 30)
        height = min(740, root.winfo_screenheight() - 65)
        root.geometry(f"{width}x{height}")
        root.minsize(min(width, 800), min(height, 560))
        root.configure(bg=BG)
        available = {f.casefold(): f for f in font.families(root)}
        self.family = next((available[f] for f in ("segoe ui", "sf pro text", "dejavu sans", "helvetica") if f in available),
                           font.nametofont("TkDefaultFont").actual("family"))
        self._style()
        self.schemas = {m: feature_schema(m) for m in MODELS.values()}
        self.suspended = True
        self.model_name = tk.StringVar(value=next(iter(MODELS)))
        self.day, self.hour, self.direction = (tk.StringVar() for _ in range(3))
        self.seed = tk.StringVar(value="2403")
        self.leads = {h: tk.BooleanVar(value=True) for h in HORIZONS}
        self.station = tk.StringVar(value="Bow Valley")
        self.seq_station = tk.StringVar(value="Calgary Intl. Airport")
        self.status = tk.StringVar()
        self.source = tk.StringVar()
        self.revision = 0
        self.names = {}
        self.fields = {}
        for model in ("spatial", "sequence-mlp"):
            for name, _, _ in self.schemas[model]:
                if name not in CALENDAR and name not in ("wind_dir_sin", "wind_dir_cos"):
                    self.fields[name] = tk.StringVar()
        self.answers = {}
        self.profile = {}
        self._layout()
        for var in [*self.fields.values(), self.day, self.hour, self.direction]:
            var.trace_add("write", self._on_edit)
        self._render_tabs()
        self.load_example()
        self.calculate()
        root.bind("<Control-Return>", lambda event: self.calculate())
        root.bind("<Configure>", self._resize)
        root.bind("<FocusIn>", self._reveal_entry)

    @property
    def model(self):
        return MODELS[self.model_name.get()]

    def _style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(self.family, 10), background=WHITE, foreground=INK)
        style.configure("TEntry", fieldbackground=WHITE, padding=4, bordercolor=LINE)
        style.configure("TCombobox", fieldbackground=WHITE, padding=4, bordercolor=LINE, arrowsize=11)
        style.map("TCombobox", fieldbackground=[("readonly", WHITE)], foreground=[("readonly", INK)])
        style.configure("TNotebook", background=WHITE, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, padding=(10, 6))
        style.map("TNotebook.Tab", background=[("selected", "#deeeeb")], foreground=[("selected", TEAL)])
        style.configure("TButton", padding=(10, 5), background=WHITE, bordercolor=LINE)
        style.configure("Action.TButton", font=(self.family, 11, "bold"), padding=(10, 9),
                        background=TEAL, foreground=WHITE, borderwidth=0)
        style.map("Action.TButton", background=[("pressed", "#004d46"), ("active", "#005d54")])
        style.configure("TCheckbutton", background=WHITE)
        style.map("TCheckbutton", background=[("active", WHITE)])

    def label(self, parent, text="", size=10, color=INK, bold=False, bg=WHITE, **options):
        return tk.Label(parent, text=text, bg=bg, fg=color, anchor="w", justify="left",
                        font=(self.family, size, "bold" if bold else "normal"), **options)

    def _layout(self):
        header = tk.Frame(self.root, bg=NAVY, padx=23, pady=12)
        header.pack(fill="x")
        self.label(header, "Calgary airport forecasts", 21, WHITE, True, NAVY).pack(side="left")
        self.label(header, "YYC   /   Temperature (°C)", 10, "#b6d9d4", bg=NAVY).pack(side="right")
        body = tk.Frame(self.root, bg=BG, padx=14, pady=14)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=WHITE, padx=14, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(4, weight=1)
        top = tk.Frame(left, bg=WHITE)
        top.grid(row=0, column=0, sticky="ew")
        self.label(top, "Observations", 16, bold=True).pack(side="left")
        ttk.Button(top, text="Load example", command=self.load_example).pack(side="right")
        ttk.Button(top, text="Clear", command=self.clear).pack(side="right", padx=5)
        self.label(left, textvariable=self.source, size=9, color=SOFT).grid(row=1, column=0, sticky="w", pady=(2, 7))
        controls = tk.Frame(left, bg=WHITE)
        controls.grid(row=2, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        for i, name in enumerate(("Model", "Issue date (YYYY-MM-DD)", "Hour (LST)")):
            self.label(controls, name, 9, SOFT).grid(row=0, column=i, sticky="w")
        self.model_box = ttk.Combobox(controls, textvariable=self.model_name, values=tuple(MODELS),
                                      state="readonly", width=21)
        self.model_box.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.model_box.bind("<<ComboboxSelected>>", self._change_model)
        ttk.Entry(controls, textvariable=self.day, width=15).grid(row=1, column=1, padx=(0, 8), sticky="ew")
        ttk.Combobox(controls, textvariable=self.hour, values=[f"{h:02}" for h in range(24)], width=4,
                     state="readonly").grid(row=1, column=2, sticky="ew")
        time_row = tk.Frame(left, bg=WHITE)
        time_row.grid(row=3, column=0, sticky="ew", pady=(5, 8))
        self.label(time_row, "Mountain Standard Time (UTC−7)", 9, SOFT).pack(side="left")
        self.live_button = ttk.Button(time_row, text="Load latest YYC", command=self.import_latest)
        self.live_button.pack(side="right")
        self.tabs = ttk.Notebook(left)
        self.tabs.grid(row=4, column=0, sticky="nsew")

        panel = tk.Frame(body, bg=WHITE, padx=14, pady=12)
        panel.grid(row=0, column=1, sticky="ns")
        self.label(panel, "Forecast leads", 15, bold=True).pack(anchor="w")
        choices = tk.Frame(panel, bg=WHITE)
        choices.pack(anchor="w", pady=(5, 10))
        self.lead_buttons = {}
        for i, (h, var) in enumerate(self.leads.items()):
            button = ttk.Checkbutton(choices, text=f"{h} h", variable=var, command=self._invalidate)
            button.grid(row=i // 3, column=i % 3, sticky="w", padx=(0, 5), pady=1)
            self.lead_buttons[h] = button
        self.seed_row = tk.Frame(panel, bg=WHITE)
        self.label(self.seed_row, "Checkpoint seed", 9, SOFT).pack(side="left", padx=(0, 6))
        seed_box = ttk.Combobox(self.seed_row, width=6, textvariable=self.seed, values=SEEDS, state="readonly")
        seed_box.pack(side="left")
        seed_box.bind("<<ComboboxSelected>>", self._invalidate)
        self.run_button = ttk.Button(panel, text="Calculate forecast", style="Action.TButton", command=self.calculate)
        self.run_button.pack(fill="x", pady=(4, 12))
        ttk.Separator(panel).pack(fill="x")
        self.label(panel, "Predicted temperature", 13, bold=True).pack(anchor="w", pady=(12, 2))
        self.label(panel, "Valid time is local standard time.", 9, SOFT).pack(anchor="w")
        results = tk.Frame(panel, bg=WHITE)
        results.pack(fill="x", pady=(8, 0))
        for col, title in enumerate(("Lead", "Valid time", "°C")):
            self.label(results, title, 9, SOFT).grid(row=0, column=col, sticky="w", padx=(0, 7))
        for i, h in enumerate(HORIZONS, 1):
            self.label(results, f"+{h} h", 10).grid(row=i, column=0, sticky="w", pady=3, padx=(0, 8))
            valid, temp = tk.StringVar(value="—"), tk.StringVar(value="—")
            self.label(results, textvariable=valid, size=9, width=13).grid(row=i, column=1, sticky="w")
            self.label(results, textvariable=temp, size=11, color=TEAL, bold=True, width=7).grid(row=i, column=2, sticky="e")
            self.answers[h] = (valid, temp)
        self.chart = tk.Canvas(panel, bg=WHITE, height=104, width=175, highlightthickness=0, takefocus=0)
        self.chart.bind("<Configure>", self._draw_profile)
        self.feedback = self.label(panel, textvariable=self.status, size=9, color=SOFT, wraplength=195)
        self.feedback.pack(fill="x", anchor="w", pady=(12, 0))

    def _resize(self, event):
        if event.widget is self.root:
            self._fit_chart()

    def _fit_chart(self):
        show = len(self.profile) >= 2 and self.root.winfo_height() >= 670
        if show and not self.chart.winfo_manager():
            self.chart.pack(fill="x", pady=(4, 0))
        elif not show and self.chart.winfo_manager():
            self.chart.pack_forget()

    def _reveal_entry(self, event):
        widget = event.widget
        canvas = self.canvases.get(self.tabs.select())
        if not isinstance(widget, ttk.Entry) or not canvas or not str(widget).startswith(str(canvas) + "."):
            return
        height = canvas.winfo_height()
        top = widget.winfo_rooty() - canvas.winfo_rooty()
        bottom = top + widget.winfo_height()
        offset = min(0, top) if top < 0 else max(0, bottom - height)
        bounds = canvas.bbox("all")
        if offset and bounds and bounds[3] > height:
            canvas.yview_moveto((canvas.canvasy(0) + offset) / bounds[3])

    def _draw_profile(self, *_):
        self.chart.delete("all")
        if len(self.profile) < 2:
            return
        left, right, top, bottom = 43, max(160, self.chart.winfo_width()) - 17, 17, 76
        leads = sorted(self.profile)
        low, high = min(self.profile.values()), max(self.profile.values())
        if high - low < 1:
            middle = (high + low) / 2
            low, high = middle - 0.5, middle + 0.5
        small = (self.family, 8)
        for y, val in [(top, high), (bottom, low)]:
            self.chart.create_line(left, y, right, y, fill=LINE)
            self.chart.create_text(left - 4, y, anchor="e", text=f"{val:.1f}°", fill=SOFT, font=small)
        points = []
        for h in leads:
            x = left + (right - left) * (h - leads[0]) / (leads[-1] - leads[0])
            y = bottom - (bottom - top) * (self.profile[h] - low) / (high - low)
            points.append((x, y))
        self.chart.create_line(*[v for pt in points for v in pt], fill=TEAL, width=2)
        for x, y in points:
            self.chart.create_oval(x-3, y-3, x+3, y+3, fill=TEAL, outline=WHITE)
        for x, lead in [(left, leads[0]), (right, leads[-1])]:
            self.chart.create_text(x, bottom+13, text=f"{lead} h", fill=SOFT, font=small)

    def _invalidate(self, *_):
        if self.suspended:
            return
        self.profile = {}
        self._fit_chart()
        self._draw_profile()
        for valid, temperature in self.answers.values():
            valid.set("—")
            temperature.set("—")
        self.feedback.configure(fg=SOFT)
        self.status.set("Choose Calculate forecast to update the results.")

    def _on_edit(self, *_):
        if not self.suspended:
            self.revision += 1
            self.source.set("Manual observations")
            self._invalidate()

    def _change_model(self, *_):
        self.revision += 1
        self._render_tabs()
        self._invalidate()

    def _scroll_page(self, title):
        page = tk.Frame(self.tabs, bg=WHITE)
        self.tabs.add(page, text=title)
        canvas = tk.Canvas(page, highlightthickness=0, bg=WHITE, width=1, height=1)
        scroll = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=WHITE, padx=8, pady=7)
        window = canvas.create_window(0, 0, anchor="nw", window=inner)
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        # Wheel events from any descendant go through this window's bind tag.
        self.canvases[str(page)] = canvas
        return inner

    def _wheel(self, event):
        canvas = self.canvases.get(self.tabs.select())
        if canvas and str(event.widget).startswith(str(canvas.master)):
            direction = -1 if getattr(event, "num", 0) == 4 or getattr(event, "delta", 0) > 0 else 1
            canvas.yview_scroll(direction * 3, "units")
            return "break"

    def _field(self, frame, row, column, name, label):
        self.names[name] = label
        self.label(frame, label, 9, SOFT).grid(row=2 * row, column=column, sticky="w", padx=(0, 9))
        entry = ttk.Entry(frame, textvariable=self.fields[name], width=11)
        entry.grid(row=2 * row + 1, column=column, sticky="ew", padx=(0, 9), pady=(0, 5))
        frame.columnconfigure(column, weight=1)

    def _render_tabs(self):
        for page in self.tabs.winfo_children():
            page.destroy()
        self.canvases = {}
        for trigger in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(trigger, self._wheel)
        self.seed_row.pack_forget()
        if self.model.startswith("sequence-"):
            self.seed_row.pack(before=self.run_button, fill="x", pady=(0, 6))
            page = self._scroll_page("Station histories")
            self._station_editor(page, True)
            return

        page = self._scroll_page("Airport weather")
        self.label(page, "Current conditions", 12, bold=True).pack(anchor="w", pady=(2, 5))
        current = tk.Frame(page, bg=WHITE)
        current.pack(fill="x")
        for i, (name, title) in enumerate({"temp_lag_0": "Current temperature (°C)", **WEATHER}.items()):
            self._field(current, i // 2, i % 2, name, title)
        self.label(current, "Wind direction (° from north)", 9, SOFT).grid(row=4, column=1, sticky="w")
        ttk.Entry(current, textvariable=self.direction, width=12).grid(row=5, column=1, sticky="ew", padx=(0, 9))
        self.label(page, "For calm wind, enter speed 0 and leave direction blank.", 9, SOFT).pack(anchor="w", pady=5)
        page = self._scroll_page("Temperature history")
        self.label(page, "Airport temperature (°C)", 12, bold=True).pack(anchor="w", pady=(2, 6))
        grid = tk.Frame(page, bg=WHITE)
        grid.pack(fill="x")
        for lag in range(24):
            name = "Current hour" if lag == 0 else f"{lag} h ago"
            self._field(grid, lag // 4, lag % 4, f"temp_lag_{lag}", name)
        if self.model == "spatial":
            self._station_editor(self._scroll_page("Regional stations"), False)

    def _station_editor(self, page, sequence):
        stations = list(STATIONS) if sequence else list(STATIONS)[1:]
        selected = self.seq_station if sequence else self.station
        menu = ttk.Combobox(page, textvariable=selected, values=stations, state="readonly", width=25)
        menu.pack(anchor="w", pady=(2, 4))
        help_text = ("Enter readings oldest to newest. The current YYC temperature is required." if sequence
                     else "Leave an unavailable reading blank to use its training median.")
        self.label(page, help_text, 9, SOFT, wraplength=415).pack(anchor="w", pady=(0, 7))
        grid = tk.Frame(page, bg=WHITE)
        grid.pack(fill="x")

        def show(*_):
            for child in grid.winfo_children():
                child.destroy()
            slug = STATIONS[selected.get()]
            if sequence:
                channels = ("temperature", "dew_point", "relative_humidity", "wind_speed")
                for j, title in enumerate(("Lag", "Temp. °C", "Dew pt. °C", "RH %", "Wind km/h")):
                    self.label(grid, title, 9, SOFT).grid(row=0, column=j, sticky="w", padx=2)
                for i, lag in enumerate(range(23, -1, -1), 1):
                    self.label(grid, f"{lag} h", 9).grid(row=i, column=0, sticky="w")
                    for j, channel in enumerate(channels, 1):
                        key = f"lag_{lag}__{slug}__{channel}"
                        grid.columnconfigure(j, weight=1)
                        ttk.Entry(grid, textvariable=self.fields[key], width=7).grid(row=i, column=j, sticky="ew", padx=2, pady=1)
            else:
                suffixes = {**{f"temp_lag_{h}": f"Temperature (°C), {h} h ago" for h in (0, 3, 6)}, **WEATHER}
                available = [(f"{slug}_{s}", label) for s, label in suffixes.items() if f"{slug}_{s}" in self.fields]
                for i, (name, label) in enumerate(available):
                    self._field(grid, i // 2, i % 2, name, label)
            canvas = self.canvases.get(self.tabs.select())
            if canvas:
                canvas.yview_moveto(0)

        menu.bind("<<ComboboxSelected>>", show)
        show()

    def import_latest(self):
        revision, model = self.revision, self.model
        mailbox = Queue()
        self.live_button.configure(text="Loading…", state="disabled")

        def fetch():
            try:
                mailbox.put(load_latest(model))
            except Exception as exc:
                mailbox.put(exc)

        def finish():
            try:
                snapshot = mailbox.get_nowait()
            except Empty:
                self.root.after(100, finish)
                return
            self.live_button.configure(text="Load latest YYC", state="normal")
            if revision != self.revision:
                self.status.set("Form changed during import. Click Load latest YYC again to replace the inputs.")
                return
            if isinstance(snapshot, Exception):
                self.feedback.configure(fg=RED)
                self.status.set(f"Import failed. {snapshot}")
                return
            self.apply_snapshot(snapshot)

        Thread(target=fetch, daemon=True).start()
        self.root.after(100, finish)

    def apply_snapshot(self, snapshot):
        self.revision += 1
        self.suspended = True
        try:
            for name, var in self.fields.items():
                value = snapshot.fields.get(name)
                var.set("" if value is None else f"{value:.10g}")
            self.direction.set("" if snapshot.direction is None else str(snapshot.direction))
            self.day.set(snapshot.origin.strftime("%Y-%m-%d"))
            self.hour.set(snapshot.origin.strftime("%H"))
            age = int((datetime.now(MST) - snapshot.origin).total_seconds() // 60)
            self.source.set(f"ECCC · {snapshot.origin:%d %b %H:%M} MST · {age} min old at import"
                            f" · Regional temperatures: {snapshot.coverage}/5")
        finally:
            self.suspended = False
        self.calculate()
        if snapshot.metar_hours:
            self.status.set(self.status.get() + f" CYYC METAR supplied {snapshot.metar_hours} temperature reading(s).")
        if snapshot.missing_stations and self.model not in ("local", "standalone-local"):
            names = ", ".join(slug.replace("_", " ").title() for slug in snapshot.missing_stations)
            self.status.set(self.status.get() + " Missing station histories: " + names + ".")

    def load_example(self):
        self.revision += 1
        self.suspended = True
        try:
            def load(name):
                return json.loads((ROOT / "examples" / (name + ".json")).read_text(encoding="utf-8"))
            spatial, sequence = load("spatial"), load("sequence")
            for name, value in spatial.items():
                if name in self.fields:
                    self.fields[name].set("" if value is None else str(value))
            for i, row in enumerate(sequence["history"]):
                for (name, _, _), value in zip(self.schemas["sequence-mlp"][i * 24:(i + 1) * 24], row):
                    self.fields[name].set("" if value is None else str(value))
            self.direction.set(f"{math.degrees(math.atan2(spatial['wind_dir_sin'], spatial['wind_dir_cos'])) % 360:.8g}")
            self.day.set("2025-01-01")
            self.hour.set("00")
            self.source.set("Historical example  ·  1 January 2025, 00:00 LST")
        finally:
            self.suspended = False
        self._invalidate()

    def clear(self):
        self.revision += 1
        self.suspended = True
        for var in (*self.fields.values(), self.day, self.hour, self.direction):
            var.set("")
        self.source.set("Manual observations")
        self.suspended = False
        self._invalidate()

    def calculate(self):
        self._invalidate()
        try:
            selected = [h for h, var in self.leads.items() if var.get()]
            if not selected:
                raise ValueError("Select at least one forecast lead.")
            origin, vector = origin_and_calendar(self.day.get(), self.hour.get())
            sequence = self.model.startswith("sequence-")
            if not sequence:
                speed = reading(self.fields["wind_speed"].get(), "airport wind speed")
                vector["wind_dir_sin"], vector["wind_dir_cos"] = wind_components(self.direction.get(), speed)
            for name, _, optional in self.schemas[self.model]:
                if name not in vector:
                    label = self.names.get(name, name).replace("_", " ")
                    value = reading(self.fields[name].get(), label, optional)
                    if value is not None:
                        if name.endswith("relative_humidity") and not 0 <= value <= 100:
                            raise ValueError(f"{label} must be between 0 and 100 percent.")
                        if name.endswith("wind_speed") and value < 0:
                            raise ValueError(f"{label} must be at least zero.")
                        if name.endswith("station_pressure") and value <= 0:
                            raise ValueError(f"{label} must be positive.")
                    vector[name] = value
            results = predict_temperature(vector, model=self.model, horizons=selected,
                                          seed=int(self.seed.get()) if sequence else None)
            valid_times = {h: (origin + timedelta(hours=h)).strftime("%d %b %H:%M") for h in results}
        except (ValueError, OSError, OverflowError) as exc:
            self.feedback.configure(fg=RED)
            self.status.set(str(exc))
            return
        for h, value in results.items():
            valid, temperature = self.answers[h]
            valid.set(valid_times[h])
            temperature.set(f"{value:.3f}")
        self.profile = results
        self._fit_chart()
        self._draw_profile()
        self.feedback.configure(fg=TEAL)
        self.status.set(f"{len(results)} forecast{'s' if len(results) != 1 else ''} calculated.")


def main():
    root = tk.Tk()
    try:
        ForecastWindow(root)
    except (OSError, ValueError, KeyError) as exc:
        messagebox.showerror("Unable to open forecast", str(exc), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
