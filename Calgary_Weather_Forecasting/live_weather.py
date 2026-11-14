"""Import hourly station observations from ECCC's public GeoMet SWOB feed."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ENDPOINT = "https://api.weather.gc.ca/collections/swob-realtime/items"
MST = timezone(timedelta(hours=-7))
HOUR = timedelta(hours=1)
MAX_AGE = timedelta(hours=3)
STATION_IDS = {
    "yyc": "3031092", "bow_valley": "3050778", "sundre": "3026KNQ",
    "drumheller": "30221LG", "claresholm": "3031640", "lacombe": "3023722",
}
# SWOB observations carry a unit alongside each measurement.
MEASUREMENTS = {
    "temperature": ("air_temp", "°C", 1),
    "dew_point": ("dwpt_temp", "°C", 1),
    "relative_humidity": ("rel_hum", "%", 1),
    "wind_speed": ("avg_wnd_spd_10m_pst2mts", "km/h", 1),
    "direction": ("avg_wnd_dir_10m_pst2mts", "°", 1),
    "station_pressure": ("stn_pres", "hPa", 0.1),
}
SEQUENCE = ("temperature", "dew_point", "relative_humidity", "wind_speed")
WEATHER = ("dew_point", "relative_humidity", "wind_speed", "station_pressure")


def parse_reports(features, station_id, start, end):
    rows = {}
    for feature in features:
        props = feature["properties"]
        if props.get("clim_id-value") != station_id:
            continue
        stamp = datetime.fromisoformat(props["obs_date_tm"].replace("Z", "+00:00"))
        if stamp.tzinfo is None or not start <= stamp <= end or stamp.minute or stamp.second or stamp.microsecond:
            continue
        values = {}
        for name, (field, unit, scale) in MEASUREMENTS.items():
            raw = props.get(field)
            if props.get(field + "-uom") != unit or type(raw) not in (float, int) or not math.isfinite(raw):
                continue
            if name == "relative_humidity" and not 0 <= raw <= 100:
                continue
            if name == "direction" and not 0 <= raw <= 360:
                continue
            if (name == "wind_speed" and raw < 0) or (name == "station_pressure" and raw <= 0):
                continue
            values[name] = raw * scale
        # Combine complementary hour reports, preferring the most complete report on conflicts.
        rows.setdefault(stamp, []).append((len(values), str(feature.get("id", "")), values))
    return {t: {k: v for _, _, record in sorted(records, key=lambda r: r[:2]) for k, v in record.items()}
            for t, records in rows.items()}


def fetch_station(station_id, start, end):
    fields = ["clim_id-value", "obs_date_tm"]
    for key, _, _ in MEASUREMENTS.values():
        fields.extend((key, key + "-uom"))
    interval = "/".join(t.strftime("%Y-%m-%dT%H:%M:%SZ") for t in (start, end))
    params = {"clim_id-value": station_id, "datetime": interval, "limit": 1000,
              "properties": ",".join(fields), "f": "json"}
    request = Request(ENDPOINT + "?" + urlencode(params), headers={"User-Agent": "CalgaryForecastGUI/1.0"})
    with urlopen(request, timeout=18) as stream:
        result = json.load(stream)
    if any(link.get("rel") == "next" for link in result.get("links", [])):
        # Some stations report every minute. Narrow the interval until each response is complete.
        if end - start <= HOUR:
            raise ValueError("ECCC returned a partial station history.")
        midpoint = start + ((end - start) // HOUR // 2) * HOUR
        return {**fetch_station(station_id, start, midpoint), **fetch_station(station_id, midpoint, end)}
    return parse_reports(result["features"], station_id, start, end)


@dataclass
class Snapshot:
    origin: datetime
    fields: dict
    direction: float | None
    coverage: int
    missing_stations: tuple
    metar_hours: int


def metar_temperatures():
    url = "https://aviationweather.gov/api/data/metar?ids=CYYC&format=json&hours=30"
    with urlopen(Request(url, headers={"User-Agent": "CalgaryForecastGUI/1.0"}), timeout=12) as stream:
        records = json.load(stream)
    values = {}
    for record in records:
        if record.get("icaoId") != "CYYC" or record.get("metarType") != "METAR":
            continue
        temp, timestamp = record.get("temp"), record.get("obsTime")
        if type(temp) not in (int, float) or not math.isfinite(temp) or type(timestamp) not in (int, float):
            continue
        time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if time.minute == time.second == time.microsecond == 0:
            values[time] = temp
    return values


def assemble(reports, now, model, fallback=None):
    fallback = fallback or {}
    airport = reports.get("yyc", {})
    for t in sorted(airport, reverse=True):
        if not timedelta(0) <= now - t <= MAX_AGE:
            continue
        current = airport[t]
        if "temperature" not in current:
            continue
        if model.startswith("sequence-"):
            break
        conditions = all(k in current for k in WEATHER) and ("direction" in current or current["wind_speed"] == 0)
        if conditions and all("temperature" in airport.get(t - i * HOUR, {}) or t - i * HOUR in fallback for i in range(24)):
            break
    else:
        raise ValueError("No usable YYC observation within the last 3 hours for this model. "
                         "The feed may be delayed or required readings may be missing. A sequence model accepts gaps in the history.")

    fields = {}
    metar_count = 0
    for slug in STATION_IDS:
        history = reports.get(slug, {})
        prefix = "" if slug == "yyc" else slug + "_"
        for lag in range(24):
            row = history.get(t - lag * HOUR, {})
            if slug == "yyc" and "temperature" not in row and t - lag * HOUR in fallback:
                row = dict(row, temperature=fallback[t - lag * HOUR])
                metar_count += 1
            for channel in SEQUENCE:
                fields[f"lag_{lag}__{slug}__{channel}"] = row.get(channel)
            if slug == "yyc" or lag in (0, 3, 6):
                fields[f"{prefix}temp_lag_{lag}"] = row.get("temperature")
        for channel in WEATHER:
            fields[prefix + channel] = history.get(t, {}).get(channel)
    missing = tuple(slug for slug in STATION_IDS if slug != "yyc" and not reports.get(slug))
    count = sum("temperature" in reports.get(slug, {}).get(t, {}) for slug in STATION_IDS if slug != "yyc")
    return Snapshot(t.astimezone(MST), fields, current.get("direction"), count, missing, metar_count)


def load_latest(model, now=None):
    now = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = end - 26 * HOUR
    reports = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        tasks = {slug: pool.submit(fetch_station, ident, start, end) for slug, ident in STATION_IDS.items()}
        for slug, task in tasks.items():
            try:
                reports[slug] = task.result()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                if slug == "yyc":
                    raise ValueError(f"Could not import YYC observations from ECCC: {exc}") from exc
                reports[slug] = {}
    fallback = {}
    recent = [t for t, row in reports["yyc"].items() if "temperature" in row and now - t <= MAX_AGE]
    if recent and any("temperature" not in reports["yyc"].get(max(recent) - i * HOUR, {}) for i in range(24)):
        try:
            fallback = metar_temperatures()
        except (OSError, ValueError, TypeError, KeyError):
            pass  # The airport or sequence model can still run if its required readings are present.
    return assemble(reports, now, model, fallback)
