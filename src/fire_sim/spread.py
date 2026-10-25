import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import shapely
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
           (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
FOUR = ndimage.generate_binary_structure(2, 1)
MIN_PROGRESS = 1e-6


@dataclass(frozen=True)
class Raster:
    x0: float
    y0: float
    res: float
    nx: int
    ny: int

    @classmethod
    def covering(cls, bounds, res, pad):
        xmin, ymin, xmax, ymax = bounds
        x0 = math.floor((xmin - pad) / res) * res
        y0 = math.floor((ymin - pad) / res) * res
        return cls(x0, y0, res, math.ceil((xmax + pad - x0) / res), math.ceil((ymax + pad - y0) / res))

    @property
    def shape(self):
        return self.ny, self.nx

    @property
    def bounds(self):
        return self.x0, self.y0, self.x0 + self.nx * self.res, self.y0 + self.ny * self.res

    def centres(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def mask(self, geometry):
        x, y = self.centres()
        shapely.prepare(geometry)
        return shapely.contains_xy(geometry, x, y)

    def top_down_transform(self):
        from affine import Affine
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0 + self.ny * self.res)


@dataclass(frozen=True)
class SpreadModel:
    wind_adjustment: float = 0.4
    slope_coefficient: float = 21.0
    max_length_to_breadth: float = 8.0
    noise_sigma: float = 0.3
    noise_length_m: float = 500.0

    def length_to_breadth(self, wind_10m_mps):
        mph = wind_10m_mps * self.wind_adjustment * 2.236936
        lb = 0.936 * math.exp(0.2566 * mph) + 0.461 * math.exp(-0.1548 * mph) - 0.397
        return min(max(lb, 1.0), self.max_length_to_breadth)

    def wind_factor(self, wind_10m_mps, cos_angle):
        lb = self.length_to_breadth(wind_10m_mps)
        e = math.sqrt(1 - 1 / lb ** 2)
        return (1 - e) / (1 - e * cos_angle)

    def slope_factor(self, rise_over_run):
        return 1 + self.slope_coefficient * np.clip(rise_over_run, 0.0, 1.0) ** 2

    def heterogeneity(self, shape, res, seed):
        if self.noise_sigma <= 0:
            return np.ones(shape)
        rng = np.random.default_rng(seed)
        z = ndimage.gaussian_filter(rng.standard_normal(shape), self.noise_length_m / res, mode="reflect")
        z /= z.std() or 1.0
        return np.exp(self.noise_sigma * z - self.noise_sigma ** 2 / 2)


@dataclass(frozen=True)
class IntervalDrivers:
    wind_to: tuple[float, float] = (0.0, 0.0)
    wind_mps: float = 0.0
    pace_times_s: tuple[float, ...] = ()
    pace_fraction: tuple[float, ...] = ()


def _travel_graph(allowed, raster, model, drivers, elevation, slowness):
    ny, nx = allowed.shape
    index = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, costs = [], [], []
    wx, wy = drivers.wind_to
    for dx, dy in OFFSETS:
        src = (slice(max(0, -dy), ny - max(0, dy)), slice(max(0, -dx), nx - max(0, dx)))
        dst = (slice(max(0, dy), ny + min(0, dy)), slice(max(0, dx), nx + min(0, dx)))
        ok = allowed[src] & allowed[dst]
        if not ok.any():
            continue
        norm = math.hypot(dx, dy)
        length = raster.res * norm
        rate = model.wind_factor(drivers.wind_mps, (dx * wx + dy * wy) / norm)
        cost = length / rate * 0.5 * (slowness[src] + slowness[dst])
        if elevation is not None:
            cost = cost / model.slope_factor((elevation[dst] - elevation[src]) / length)
        rows.append(index[src][ok])
        cols.append(index[dst][ok])
        costs.append(cost[ok])
    n = ny * nx
    if not rows:
        return csr_matrix((n, n))
    return csr_matrix((np.concatenate(costs), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def _with_terminal(graph, cells, costs, outgoing):
    n = graph.shape[0]
    coo = graph.tocoo()
    terminal = np.full(len(cells), n)
    rows, cols = (terminal, cells) if outgoing else (cells, terminal)
    return csr_matrix((np.concatenate([coo.data, costs]),
                       (np.concatenate([coo.row, rows]), np.concatenate([coo.col, cols]))),
                      shape=(n + 1, n + 1)), n


def _edge_costs(raster, mask, geometry, slowness):
    x, y = raster.centres()
    distance = shapely.distance(geometry.boundary, shapely.points(x[mask], y[mask]))
    return np.maximum(distance, 1e-3) * slowness[mask]


def _progress(before, after, band_graph, raster, before_geom, after_geom, slowness):
    band = after & ~before
    shape = before.shape
    size = before.size
    entry = band & ndimage.binary_dilation(before, np.ones((3, 3)))
    forward = np.full(size, np.inf)
    if entry.any():
        graph, source = _with_terminal(band_graph, np.flatnonzero(entry),
                                       _edge_costs(raster, entry, before_geom, slowness), outgoing=True)
        forward = dijkstra(graph, indices=source)[:size]
    forward = forward.reshape(shape)
    unreached = band & ~np.isfinite(forward)
    spots = 0
    if unreached.any():
        labels, spots = ndimage.label(unreached, structure=np.ones((3, 3)))
        distance = (ndimage.distance_transform_edt(~before) if before.any()
                    else np.zeros(shape))
        ignitions = ndimage.minimum_position(distance, labels, range(1, spots + 1))
        flat = np.ravel_multi_index(np.array(ignitions).T, shape)
        spot_time = dijkstra(band_graph, indices=flat, min_only=True).reshape(shape)
        forward = np.where(unreached, spot_time, forward)
    exits = band & ndimage.binary_dilation(~after, FOUR, border_value=1)
    backward = np.full(size, np.inf)
    if exits.any():
        graph, sink = _with_terminal(band_graph, np.flatnonzero(exits),
                                     _edge_costs(raster, exits, after_geom, slowness), outgoing=False)
        backward = dijkstra(graph.T.tocsr(), indices=sink)[:size]
    backward = backward.reshape(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = forward / (forward + backward)
    enclosed = band & ~np.isfinite(backward)
    islands = 0
    if enclosed.any():
        labels, islands = ndimage.label(enclosed, structure=np.ones((3, 3)))
        peak = np.asarray(ndimage.maximum(forward, labels, range(1, islands + 1)))
        fraction = np.where(enclosed, forward / np.maximum(peak, 1e-12)[np.maximum(labels - 1, 0)], fraction)
    fraction = np.where(band, np.clip(np.nan_to_num(fraction, nan=1.0, posinf=1.0), MIN_PROGRESS, 1.0), np.nan)
    return fraction, {"band_cells": int(band.sum()), "spot_fires": int(spots), "enclosed_islands": int(islands)}


def interpolate(raster, geometries, times_s, *, model=SpreadModel(), drivers=None,
                elevation=None, seed=0):
    if len(geometries) != len(times_s) or not geometries:
        raise ValueError("Need one time per perimeter")
    if any(b <= a for a, b in zip(times_s, times_s[1:])):
        raise ValueError("Snapshot times must strictly increase")
    drivers = drivers or [IntervalDrivers()] * (len(times_s) - 1)
    slowness = 1 / model.heterogeneity(raster.shape, raster.res, seed)
    burned = raster.mask(geometries[0])
    burned_geom = geometries[0]
    arrival = np.where(burned, float(times_s[0]), np.inf)
    diagnostics = []
    for k in range(len(times_s) - 1):
        observed = raster.mask(geometries[k + 1])
        after = observed | burned
        after_geom = shapely.union(burned_geom, geometries[k + 1])
        d = drivers[k]
        band_graph = _travel_graph(after & ~burned, raster, model, d, elevation, slowness)
        fraction, info = _progress(burned, after, band_graph, raster, burned_geom, after_geom, slowness)
        span = times_s[k + 1] - times_s[k]
        if d.pace_times_s:
            clock = np.interp(fraction, d.pace_fraction, d.pace_times_s)
        else:
            clock = times_s[k] + fraction * span
        band = after & ~burned
        arrival[band] = clock[band]
        diagnostics.append({"interval": k, "start_s": times_s[k], "end_s": times_s[k + 1],
                            "clipped_shrink_cells": int((burned & ~observed).sum()),
                            "wind_mps": d.wind_mps, "wind_to": list(d.wind_to),
                            "length_to_breadth": model.length_to_breadth(d.wind_mps), **info})
        burned, burned_geom = after, after_geom
    return arrival, diagnostics


def spread_rate(arrival, res, initial):
    valid = np.isfinite(arrival) & ~initial
    t = np.where(valid, arrival, np.nan)
    grads = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis in (0, 1):
            fwd = np.diff(t, axis=axis, append=np.nan)
            bwd = np.diff(t, axis=axis, prepend=np.nan)
            both = np.stack([fwd, bwd])
            count = np.isfinite(both).sum(axis=0)
            grads.append(np.where(count > 0, np.nansum(both, axis=0) / np.maximum(count, 1), np.nan) / res)
        rate = 1 / np.hypot(*grads)
    rate[~valid | ~np.isfinite(rate)] = np.nan
    return rate


@dataclass
class ArrivalField:
    raster: Raster
    crs: str
    t0: datetime
    arrival_s: np.ndarray
    snapshot_s: tuple[float, ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.initial = self.arrival_s <= self.snapshot_s[0]
        self.rate_mps = spread_rate(self.arrival_s, self.raster.res, self.initial)

    @property
    def start_s(self):
        return self.snapshot_s[0]

    @property
    def end_s(self):
        return self.snapshot_s[-1]

    def burned(self, t):
        return self.arrival_s <= t

    def next_snapshot(self, t):
        later = [s for s in self.snapshot_s if s > t]
        return later[0] if later else self.end_s

    def perimeter(self, t):
        from rasterio import features
        mask = np.flipud(self.burned(t)).astype(np.uint8)
        shapes = [shapely.geometry.shape(g) for g, v in
                  features.shapes(mask, mask=mask.astype(bool), transform=self.raster.top_down_transform())]
        return shapely.union_all(shapes) if shapes else shapely.Polygon()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), arrival_s=self.arrival_s)
        path.with_suffix(".json").write_text(json.dumps({
            "raster": asdict(self.raster), "crs": self.crs, "t0": self.t0.isoformat(),
            "snapshot_s": list(self.snapshot_s), "meta": self.meta}, indent=2, default=str) + "\n")

    @classmethod
    def load(cls, path):
        path = Path(path)
        info = json.loads(path.with_suffix(".json").read_text())
        arrival = np.load(path.with_suffix(".npz"))["arrival_s"]
        return cls(Raster(**info["raster"]), info["crs"], datetime.fromisoformat(info["t0"]),
                   arrival, tuple(info["snapshot_s"]), info["meta"])
