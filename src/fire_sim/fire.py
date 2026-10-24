"""Build a continuous fire (ArrivalField) from a FEDS sequence plus weather and terrain."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import shapely
from pyproj import Transformer

from .feds import load_feds
from .spread import ArrivalField, Raster, SpreadModel, interpolate
from .weather import Weather, fetch_hourly, overpass_utc


def snapshot_clock(sequence, lon, convention="overpass"):
    """UTC time of each snapshot and seconds after the first."""
    if convention == "overpass":
        stamps = [overpass_utc(p.timestamp, lon) for p in sequence]
    elif convention == "label":
        from datetime import timezone
        stamps = [p.timestamp.replace(tzinfo=timezone.utc) for p in sequence]
    else:
        raise ValueError("convention must be overpass or label")
    return stamps, [(s - stamps[0]).total_seconds() for s in stamps]


@dataclass
class Inputs:
    """Everything the interpolation needs besides the spread model and seed."""
    raster: Raster
    crs: str
    stamps: list
    times_s: list
    lonlat: tuple
    elevation: object = None
    weather: object = None

    def drivers(self, indices):
        """Interval drivers between consecutive snapshot indices (or None if no weather)."""
        if self.weather is None:
            return None
        return [self.weather.drivers(self.times_s[a], self.times_s[b]) for a, b in zip(indices, indices[1:])]


def prepare_inputs(sequence, crs, *, res=100.0, pad=2000.0, weather=True, terrain=True,
                   convention="overpass", pacing_exponent=1.0, cache_dir="data/cache"):
    if len(sequence) < 2:
        raise ValueError("Need at least two snapshots to interpolate")
    centre = sequence[0].geometry.centroid
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(centre.x, centre.y)
    stamps, times_s = snapshot_clock(sequence, lon, convention)
    raster = Raster.covering(shapely.union_all([p.geometry for p in sequence]).bounds, res, pad)
    inputs = Inputs(raster, str(crs), stamps, times_s, (lon, lat))
    if terrain:
        from .terrain import fetch_elevation
        inputs.elevation = fetch_elevation(raster, crs, Path(cache_dir) / "dem")
    if weather:
        payload = fetch_hourly(lat, lon, stamps[0].date(), stamps[-1].date(), Path(cache_dir) / "weather")
        inputs.weather = Weather.from_open_meteo(payload, stamps[0], exponent=pacing_exponent)
    return inputs


def build_field(sequence, crs, *, seed=0, model=SpreadModel(), provenance=None, **input_options):
    inputs = prepare_inputs(sequence, crs, **input_options)
    arrival, diagnostics = interpolate(inputs.raster, [p.geometry for p in sequence], inputs.times_s,
                                       model=model, drivers=inputs.drivers(range(len(sequence))),
                                       elevation=inputs.elevation, seed=seed)
    weather_meta = ({"source": "Open-Meteo archive (ERA5), hourly, single point at first-perimeter centroid",
                     "pacing": "cumulative FFWI**exponent + floor", "floor": inputs.weather.floor,
                     "exponent": inputs.weather.exponent}
                    if inputs.weather is not None else {"source": "none: uniform pacing, isotropic spread"})
    meta = {"provenance": provenance or {}, "centroid_lonlat": list(inputs.lonlat), "seed": seed,
            "model": asdict(model), "time_convention": input_options.get("convention", "overpass"),
            "snapshot_labels": [p.timestamp.isoformat() for p in sequence],
            "snapshot_utc": [s.isoformat() for s in inputs.stamps], "weather": weather_meta,
            "terrain": "Copernicus GLO-30 DEM, bilinear" if inputs.elevation is not None else "none (flat)",
            "intervals": diagnostics}
    return ArrivalField(inputs.raster, inputs.crs, inputs.stamps[0], arrival, tuple(inputs.times_s), meta)


def load_fire(path, *, fire_id, year=None, limit=None, crs=None, repair=True,
              cache_dir="data/cache", **options):
    """Load FEDS and build (or reuse a cached) ArrivalField. Returns (field, sequence)."""
    sequence, provenance = load_feds(path, fire_id=fire_id, year=year, target_crs=crs,
                                     repair=repair, limit=limit)
    crs = provenance["target_crs"]
    key_parts = {"sha": provenance["source_sha256"], "fire": str(fire_id), "year": year,
                 "limit": limit, "crs": crs, **{k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v)
                                                for k, v in sorted(options.items())}}
    key = hashlib.sha256(json.dumps(key_parts, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cached = Path(cache_dir) / "fields" / f"fire{fire_id}_{key}"
    if cached.with_suffix(".npz").exists():
        return ArrivalField.load(cached), sequence
    field = build_field(sequence, crs, provenance=provenance, cache_dir=cache_dir, **options)
    field.save(cached)
    return field, sequence
