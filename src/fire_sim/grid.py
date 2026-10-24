"""Fixed world grid: boundary intersection, not burned-interior coverage."""

import hashlib
import math
from dataclasses import dataclass

from shapely.geometry import box
from shapely.prepared import prep

from .models import ChangeSet, MonitoringPoint


@dataclass(frozen=True)
class Grid:
    cell_size_m: float = 500.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    base_revisit_s: float = 900.0
    seed: int = 42
    max_candidate_cells: int = 1_000_000

    def __post_init__(self):
        if not math.isfinite(self.cell_size_m) or self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be finite and positive")
        if not math.isfinite(self.base_revisit_s) or self.base_revisit_s <= 0:
            raise ValueError("base_revisit_s must be finite and positive")
        if not all(math.isfinite(v) for v in (self.origin_x, self.origin_y)):
            raise ValueError("grid origin must be finite")

    def points(self, polygon):
        if polygon.is_empty or not polygon.is_valid or polygon.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Expected a valid nonempty Polygon or MultiPolygon")
        size = self.cell_size_m
        xmin, ymin, xmax, ymax = polygon.bounds
        # Include cells touching the boundary on either side of a grid line.
        lo_x = math.ceil((xmin - self.origin_x) / size) - 1
        lo_y = math.ceil((ymin - self.origin_y) / size) - 1
        hi_x = math.floor((xmax - self.origin_x) / size)
        hi_y = math.floor((ymax - self.origin_y) / size)
        if (hi_x - lo_x + 1) * (hi_y - lo_y + 1) > self.max_candidate_cells:
            raise ValueError("Grid is too fine; increase cell_size_m")
        boundary = prep(polygon.boundary)
        result = {}
        for ix in range(lo_x, hi_x + 1):
            for iy in range(lo_y, hi_y + 1):
                x, y = self.origin_x + ix * size, self.origin_y + iy * size
                if boundary.intersects(box(x, y, x + size, y + size)):
                    # Stable across Python processes, births, removals and reappearances.
                    digest = hashlib.sha256(f"{self.seed}:{ix}:{iy}".encode()).digest()
                    risk = 1 + int.from_bytes(digest[:8], "big") % 3
                    result[ix, iy] = MonitoringPoint((ix, iy), x + size / 2, y + size / 2,
                                                    risk, self.base_revisit_s / risk)
        return result


def changes(previous, current, *, geometry_changed=()):
    old, new = previous.keys(), current.keys()
    altered = {i for i in old & new if previous[i] != current[i]}
    altered.update(set(geometry_changed) & old & new)
    return ChangeSet(frozenset(new - old), frozenset(old - new), frozenset(altered))


def perimeter_changes(grid, before, after, previous, current):
    """Also flag surviving cells whose local boundary segment changed."""
    altered = []
    for i in previous.keys() & current.keys():
        p = current[i]
        half = grid.cell_size_m / 2
        cell = box(p.x - half, p.y - half, p.x + half, p.y + half)
        if not before.boundary.intersection(cell).equals(after.boundary.intersection(cell)):
            altered.append(i)
    return changes(previous, current, geometry_changed=altered)
