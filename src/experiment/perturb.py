"""Controlled perturbations of a PlanningState.

The proposal's two change factors are
  * magnitude: proportion of affected points
  * locality : concentrated (one contiguous stretch of perimeter) or distributed

`apply_change` takes a base state and returns a new state that differs from it in
exactly that way, so the *same* change can be replayed against every planner.

What "affected" means (`kind`)
  risk    : the point's risk tier changes, so its revisit period q changes (a "changed" point)
  advance : the point disappears and a new point appears one cell further out
            (the front moved: one "removed" + one "added")
  mixed   : alternate the two over the affected points (default)

Direction of a risk change (`direction`)
  cycle : 1 -> 2 -> 3 -> 1 (original rule; a picked risk-3 point drops two tiers, so the share of
          drops depends on how many risk-3 points the base state has)
  up    : one tier up (the fire is spreading faster there); a risk-3 point cannot go up, so it
          advances instead (the fast front leaves its cell)

The realised magnitude is measured with the same definition the repo uses
(`fire_sim.models.changes(...).fraction`): |added u removed u changed| / |previous u current|.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace

from fire_sim.models import MonitoringPoint, changes
from fire_sim.simulation import PlanningState, coverage

LOCALITIES = ("concentrated", "distributed")
KINDS = ("risk", "advance", "mixed")
DIRECTIONS = ("cycle", "up")


@dataclass(frozen=True)
class ChangeSpec:
    fraction: float  # target |added u removed u changed| / |previous u current|
    locality: str = "concentrated"
    kind: str = "mixed"
    direction: str = "cycle"

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError("fraction must be in [0, 1)")
        if self.locality not in LOCALITIES:
            raise ValueError(f"locality must be one of {LOCALITIES}")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")


@dataclass(frozen=True)
class ChangeInfo:
    target_fraction: float
    realised_fraction: float
    n_before: int
    n_after: int
    n_affected: int  # points selected for change
    added: int
    removed: int
    changed: int
    dispersion: float  # mean pairwise distance of affected points / that of all points (<1: clustered)


def _n_affected(n: int, fraction: float, kind: str) -> int:
    """Number of points to touch so that the realised fraction is closest to the target."""
    def frac(m: int) -> float:
        a = m // 2 if kind == "mixed" else (m if kind == "advance" else 0)
        r = m - a
        return (r + 2 * a) / (n + a)

    if fraction <= 0 or n == 0:
        return 0
    return min(range(1, n + 1), key=lambda m: abs(frac(m) - fraction))


def _mean_pairwise(xy: list[tuple[float, float]]) -> float:
    if len(xy) < 2:
        return 0.0
    total = count = 0
    for a in range(len(xy)):
        for b in range(a + 1, len(xy)):
            total += math.hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
            count += 1
    return total / count


def select_affected(state: PlanningState, m: int, locality: str, rng: random.Random) -> list:
    ids = sorted(state.points)
    if m >= len(ids):
        return ids
    if locality == "distributed":
        return sorted(rng.sample(ids, m))
    centre = state.points[rng.choice(ids)]
    by_distance = sorted(ids, key=lambda i: (math.hypot(state.points[i].x - centre.x, state.points[i].y - centre.y), i))
    return by_distance[:m]


def apply_change(state: PlanningState, spec: ChangeSpec, seed: int, *, cell_size_m: float,
                 sensor_radius_m: float | None = None) -> tuple[PlanningState, ChangeInfo]:
    """Return (changed state, description). Time, drones and last_seen of kept points are untouched."""
    rng = random.Random(seed)
    n = len(state.points)
    m = _n_affected(n, spec.fraction, spec.kind)
    affected = select_affected(state, m, spec.locality, rng)
    cx = sum(p.x for p in state.points.values()) / max(1, n)
    cy = sum(p.y for p in state.points.values()) / max(1, n)

    points = dict(state.points)
    last_seen = dict(state.last_seen)
    for pos, i in enumerate(affected):
        p = state.points[i]
        do_advance = spec.kind == "advance" or (spec.kind == "mixed" and pos % 2 == 1)
        if not do_advance and spec.direction == "up" and p.risk == 3:
            do_advance = True
        if do_advance:
            moved = _advance(p, cx, cy, cell_size_m, points)
            if moved is not None:
                del points[i], last_seen[i]
                points[moved.id] = moved
                last_seen[moved.id] = state.time_s  # activation is not an observation
                continue
        # risk change (also the fallback when the outward cell is taken)
        risk = min(3, p.risk + 1) if spec.direction == "up" else p.risk % 3 + 1
        base = p.revisit_s * p.risk
        points[i] = MonitoringPoint(i, p.x, p.y, risk, base / risk)

    if state.coverage is None:
        cov = None
    else:
        radius = cell_size_m if sensor_radius_m is None else sensor_radius_m
        cov = coverage(points, radius)
    new_state = replace(state, points=points, last_seen=last_seen, coverage=cov)

    delta = changes(state.points, points)
    xy_all = [(p.x, p.y) for p in state.points.values()]
    xy_aff = [(state.points[i].x, state.points[i].y) for i in affected]
    base_spread = _mean_pairwise(xy_all)
    info = ChangeInfo(
        target_fraction=spec.fraction,
        realised_fraction=delta.fraction(state.points, points) if (delta.added or delta.removed or delta.changed) else 0.0,
        n_before=n, n_after=len(points), n_affected=len(affected),
        added=len(delta.added), removed=len(delta.removed), changed=len(delta.changed),
        dispersion=_mean_pairwise(xy_aff) / base_spread if base_spread > 0 else 0.0,
    )
    return new_state, info


def _advance(p: MonitoringPoint, cx: float, cy: float, cell: float, existing: dict) -> MonitoringPoint | None:
    """A new point one (or two) cells further from the centroid than p, in a free grid cell."""
    dx, dy = p.x - cx, p.y - cy
    norm = math.hypot(dx, dy) or 1.0
    for steps in (1, 2):
        x = p.x + dx / norm * cell * steps
        y = p.y + dy / norm * cell * steps
        cid = (math.floor(x / cell), math.floor(y / cell))
        if cid not in existing:
            return MonitoringPoint(cid, (cid[0] + 0.5) * cell, (cid[1] + 0.5) * cell, p.risk, p.revisit_s)
    return None
