import math

import numpy as np
from scipy import ndimage

from .models import MonitoringPoint
from .spread import FOUR, ArrivalField


class FireMap:
    def __init__(self, field: ArrivalField, *, cell_size_m=500.0, sensor_radius_m=500.0,
                 base_revisit_s=900.0, rate_tiers_mpm=(2.0, 10.0)):
        low, high = rate_tiers_mpm
        if not 0 <= low <= high:
            raise ValueError("rate tiers must satisfy 0 <= low <= high")
        if cell_size_m <= 0 or sensor_radius_m < 0 or base_revisit_s <= 0:
            raise ValueError("cell size and base revisit must be positive, sensor radius non-negative")
        self.field = field
        self.cell_size_m = cell_size_m
        self.sensor_radius_m = sensor_radius_m
        self.base_revisit_s = base_revisit_s
        self.rate_tiers_mpm = rate_tiers_mpm
        raster = field.raster
        self.known = np.zeros(raster.shape, bool)
        self.seen_at = np.full(raster.shape, -np.inf)
        x, y = raster.centres()
        self.cell_x = np.floor(x / cell_size_m).astype(np.int64)
        self.cell_y = np.floor(y / cell_size_m).astype(np.int64)
        self.evidence = {}
        self.snapshots_ingested = []

    def ingest_snapshot(self, k):
        field = self.field
        s = field.snapshot_s[k]
        observed = field.arrival_s <= s
        self.known |= observed
        self.seen_at = np.maximum(self.seen_at, s)
        if k > 0:
            previous = field.arrival_s <= field.snapshot_s[k - 1]
            beyond = ndimage.distance_transform_edt(~previous) * field.raster.res
            rate = np.where(observed & ~previous, beyond / (s - field.snapshot_s[k - 1]), 0.0)
            perimeter = self._perimeter()
            self._record(perimeter, rate, s)
        self.snapshots_ingested.append(k)

    def observe(self, x, y, t):
        raster = self.field.raster
        r = self.sensor_radius_m
        margin = 2 * r + raster.res
        ix0 = max(0, int((x - margin - raster.x0) // raster.res))
        ix1 = min(raster.nx, int(math.ceil((x + margin - raster.x0) / raster.res)))
        iy0 = max(0, int((y - margin - raster.y0) // raster.res))
        iy1 = min(raster.ny, int(math.ceil((y + margin - raster.y0) / raster.res)))
        if ix0 >= ix1 or iy0 >= iy1:
            return
        window = (slice(iy0, iy1), slice(ix0, ix1))
        xs = raster.x0 + (np.arange(ix0, ix1) + 0.5) * raster.res
        ys = raster.y0 + (np.arange(iy0, iy1) + 0.5) * raster.res
        disc = np.hypot(xs[None, :] - x, ys[:, None] - y) <= r
        if not disc.any():
            return
        before = self.known[window].copy()
        burned = (self.field.arrival_s[window] <= t) & disc
        newly = burned & ~before
        beyond = ndimage.distance_transform_edt(~before) * raster.res if before.any() else np.zeros(before.shape)
        elapsed = np.maximum(t - self.seen_at[window], 1.0)
        rate = np.where(newly & np.isfinite(elapsed), beyond / elapsed, 0.0)
        self.known[window] |= burned
        self.seen_at[window] = np.where(disc, t, self.seen_at[window])
        perimeter = self._perimeter(window) & disc
        self._record(perimeter | newly, rate, t, window)

    def _perimeter(self, window=(slice(None), slice(None))):
        known = self.known
        perimeter = known & ndimage.binary_dilation(~known, FOUR, border_value=0)
        return perimeter[window]

    def _record(self, mask, rate, t, window=(slice(None), slice(None))):
        if not mask.any():
            return
        keys = np.stack([self.cell_x[window][mask], self.cell_y[window][mask]], axis=1)
        cells, inverse = np.unique(keys, axis=0, return_inverse=True)
        best = np.zeros(len(cells))
        np.maximum.at(best, inverse.ravel(), rate[mask])
        for (i, j), value in zip(cells, best):
            cell = (int(i), int(j))
            if cell not in self.evidence or self.evidence[cell][1] <= t:
                self.evidence[cell] = (float(value), float(t))

    def risk(self, rate_mps):
        mpm = rate_mps * 60
        low, high = self.rate_tiers_mpm
        return 1 + int(mpm >= low) + int(mpm >= high)

    def points(self):
        perimeter = self._perimeter()
        if not perimeter.any():
            return {}
        cells = np.unique(np.stack([self.cell_x[perimeter], self.cell_y[perimeter]], axis=1), axis=0)
        size = self.cell_size_m
        result = {}
        for i, j in cells:
            cell = (int(i), int(j))
            risk = self.risk(self.evidence.get(cell, (0.0, 0.0))[0])
            result[cell] = MonitoringPoint(cell, cell[0] * size + size / 2, cell[1] * size + size / 2,
                                           risk, self.base_revisit_s / risk)
        return result

    def known_fraction(self, t):
        truth = self.field.arrival_s <= t
        return float((self.known & truth).sum() / max(1, truth.sum()))
