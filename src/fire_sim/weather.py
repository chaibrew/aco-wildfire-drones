import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .spread import IntervalDrivers

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m")


def fetch_hourly(lat, lon, start, end, cache_dir="data/cache/weather"):
    cache = Path(cache_dir) / f"{lat:.3f}_{lon:.3f}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if not cache.exists():
        query = urllib.parse.urlencode({
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "start_date": f"{start:%Y-%m-%d}", "end_date": f"{end:%Y-%m-%d}",
            "hourly": ",".join(VARIABLES), "wind_speed_unit": "ms", "timezone": "GMT"})
        with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=60) as response:
            payload = json.load(response)
        if "hourly" not in payload:
            raise ValueError(f"Open-Meteo returned no hourly data: {payload}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
    return json.loads(cache.read_text())


def equilibrium_moisture(temp_c, rh):
    t = np.asarray(temp_c) * 9 / 5 + 32
    h = np.asarray(rh, dtype=float)
    return np.select(
        [h < 10, h < 50],
        [0.03229 + 0.281073 * h - 0.000578 * h * t,
         2.22749 + 0.160107 * h - 0.01478 * t],
        21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h)


def ffwi(temp_c, rh, wind_mps):
    m = np.clip(equilibrium_moisture(temp_c, rh), 0, 30) / 30
    eta = 1 - 2 * m + 1.5 * m ** 2 - 0.5 * m ** 3
    mph = np.asarray(wind_mps) * 2.236936
    return eta * np.sqrt(1 + mph ** 2) / 0.3002


@dataclass(frozen=True)
class Weather:
    t0: datetime
    times_s: np.ndarray
    temp_c: np.ndarray
    rh: np.ndarray
    wind_mps: np.ndarray
    wind_from_deg: np.ndarray
    floor: float = 1.0
    exponent: float = 1.0

    @classmethod
    def from_open_meteo(cls, payload, t0, floor=1.0, exponent=1.0):
        hourly = payload["hourly"]
        stamps = [datetime.fromisoformat(s).replace(tzinfo=timezone.utc) for s in hourly["time"]]
        values = {k: np.array([np.nan if v is None else v for v in hourly[k]], dtype=float) for k in VARIABLES}
        if any(np.isnan(v).all() for v in values.values()):
            raise ValueError("Weather series is empty for this location/period")
        values = {k: _fill(v) for k, v in values.items()}
        return cls(t0, np.array([(s - t0).total_seconds() for s in stamps]),
                   values["temperature_2m"], values["relative_humidity_2m"],
                   values["wind_speed_10m"], values["wind_direction_10m"], floor, exponent)

    @property
    def index(self):
        return ffwi(self.temp_c, self.rh, self.wind_mps)

    def drivers(self, start_s, end_s, samples=64):
        if not (self.times_s[0] <= start_s and end_s <= self.times_s[-1]):
            raise ValueError("Weather series does not cover the snapshot interval")
        inside = self.times_s[(self.times_s > start_s) & (self.times_s < end_s)]
        clock = np.unique(np.concatenate([[start_s, end_s], inside,
                                          np.linspace(start_s, end_s, samples)]))
        weight = np.interp(clock, self.times_s, self.index) ** self.exponent + self.floor
        cumulative = np.concatenate([[0], np.cumsum(np.diff(clock) * (weight[1:] + weight[:-1]) / 2)])
        fraction = cumulative / cumulative[-1]
        rad = np.radians(self.wind_from_deg)
        east = np.interp(clock, self.times_s, -np.sin(rad) * self.wind_mps)
        north = np.interp(clock, self.times_s, -np.cos(rad) * self.wind_mps)
        mean_e, mean_n = np.average(east, weights=weight), np.average(north, weights=weight)
        magnitude = math.hypot(mean_e, mean_n)
        wind_to = (mean_e / magnitude, mean_n / magnitude) if magnitude > 1e-9 else (0.0, 0.0)
        return IntervalDrivers(tuple(map(float, wind_to)), float(magnitude),
                               tuple(map(float, clock)), tuple(map(float, fraction)))


def _fill(values):
    good = np.isfinite(values)
    if good.all():
        return values
    idx = np.arange(len(values))
    return np.interp(idx, idx[good], values[good])


def overpass_utc(label, lon):
    hours = 1.5 if label.hour < 12 else 13.5
    day = datetime(label.year, label.month, label.day, tzinfo=timezone.utc)
    return day + timedelta(hours=hours - lon / 15)
