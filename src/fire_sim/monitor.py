"""Monitoring points from a continuous fire at any instant.

A fine raster cell is on the perimeter at time t if it has burned and a
4-neighbour has not. It is on the *active front* if one of those unburned
neighbours burns before the next FEDS observation (default), or within
`active_window_s` if set; its spread rate is the fastest such neighbour's rate.
Because each observation interval has its own growth band, the active set
drifts slowly between snapshots and shifts substantially at snapshot times.
Fine cells roll up to the same fixed world grid as grid.Grid (origin 0,
square cells), so cell IDs stay stable across time.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import MonitoringPoint
from .spread import FOUR, ArrivalField


@dataclass(frozen=True)
class FrontMonitor:
    field: ArrivalField
    cell_size_m: float = 500.0
    base_revisit_s: float = 900.0
    # Spread-rate thresholds (metres/minute) for risk 2 and risk 3; q = base / risk.
    rate_tiers_mpm: tuple[float, float] = (2.0, 10.0)
    active_only: bool = True
    active_window_s: float | None = None

    def __post_init__(self):
        low, high = self.rate_tiers_mpm
        if not 0 <= low <= high:
            raise ValueError("rate tiers must satisfy 0 <= low <= high")
        if self.active_window_s is not None and self.active_window_s <= 0:
            raise ValueError("active window must be positive")
        if self.cell_size_m <= 0 or self.base_revisit_s <= 0:
            raise ValueError("cell size and base revisit must be positive")

    def _check(self, t):
        if not self.field.start_s <= t <= self.field.end_s:
            raise ValueError(f"t={t} outside observed span [{self.field.start_s}, {self.field.end_s}]")

    def front(self, t):
        """Fine-raster perimeter mask, active-front mask and front rate (m/s, 0 if inactive)."""
        self._check(t)
        arrival = self.field.arrival_s
        burned = arrival <= t
        perimeter = burned & ndimage.binary_dilation(~burned, FOUR, border_value=0)
        horizon = self.field.next_snapshot(t) if self.active_window_s is None else t + self.active_window_s
        advancing = ~burned & (arrival <= horizon)
        rate = np.where(advancing, np.nan_to_num(self.field.rate_mps, nan=0.0), -np.inf)
        reach = ndimage.maximum_filter(rate, footprint=FOUR, mode="constant", cval=-np.inf)
        active = perimeter & np.isfinite(reach)
        return perimeter, active, np.where(active, reach, 0.0)

    def risk(self, rate_mps):
        mpm = np.asarray(rate_mps) * 60
        low, high = self.rate_tiers_mpm
        return 1 + (mpm >= low).astype(int) + (mpm >= high).astype(int)

    def cell_rates(self, t):
        """{cell_id: front rate (m/s)} for monitored grid cells at time t."""
        perimeter, active, rate = self.front(t)
        selected = active if self.active_only else perimeter
        if not selected.any():
            return {}
        x, y = self.field.raster.centres()
        ix = np.floor(x[selected] / self.cell_size_m).astype(np.int64)
        iy = np.floor(y[selected] / self.cell_size_m).astype(np.int64)
        keys, inverse = np.unique(np.stack([ix, iy], axis=1), axis=0, return_inverse=True)
        best = np.zeros(len(keys))
        np.maximum.at(best, inverse.ravel(), rate[selected])
        return {(int(i), int(j)): float(r) for (i, j), r in zip(keys, best)}

    def points(self, t):
        """{cell_id: MonitoringPoint} at time t, with risk tiers from current spread rate."""
        size = self.cell_size_m
        result = {}
        for (i, j), rate in self.cell_rates(t).items():
            risk = int(self.risk(rate))
            result[i, j] = MonitoringPoint((i, j), i * size + size / 2, j * size + size / 2,
                                           risk, self.base_revisit_s / risk)
        return result
