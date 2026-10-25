"""Deterministic, event-based point-observation simulation."""

import math
from dataclasses import dataclass

import numpy as np

from .models import CellId, MonitoringPoint


@dataclass(frozen=True)
class Drone:
    id: int
    x: float
    y: float


@dataclass(frozen=True)
class PlanningState:
    time_s: float
    horizon_s: float
    speed_mps: float
    service_s: float
    drones: tuple[Drone, ...]
    points: dict[CellId, MonitoringPoint]
    last_seen: dict[CellId, float]
    coverage: dict[CellId, tuple[CellId, ...]] | None = None


@dataclass
class Evaluation:
    objective: float
    overdue_count: int
    mean_gap_s: float
    max_gap_s: float
    total_distance_m: float
    longest_route_m: float
    gaps_s: dict
    observations: dict
    end_drones: tuple
    visits: list

    def summary(self):
        return {key: getattr(self, key) for key in (
            "objective", "overdue_count", "mean_gap_s", "max_gap_s",
            "total_distance_m", "longest_route_m")}


def evaluate(state: PlanningState, routes: dict[int, list[CellId]]) -> Evaluation:
    """Pure evaluation shared by all planners; infeasible routes raise ValueError.

    Observations occur on service completion. No implicit start-position visits,
    return-to-base, range sensing, or observations during final idle time.
    """
    if set(routes) != {d.id for d in state.drones}:
        raise ValueError("Provide exactly one route (possibly empty) per drone")
    observations = {i: [] for i in state.points}
    coverage = state.coverage or {}
    distances, end_drones, visits = [], [], []
    end = state.time_s + state.horizon_s
    for drone in state.drones:
        x, y, clock, distance = drone.x, drone.y, state.time_s, 0.0
        for i in routes[drone.id]:
            if i not in state.points:
                raise ValueError(f"Route references inactive point {i}")
            point = state.points[i]
            leg = math.hypot(point.x - x, point.y - y)
            clock += leg / state.speed_mps + state.service_s
            if clock > end + 1e-9:
                raise ValueError(f"Drone {drone.id} route exceeds horizon")
            for j in coverage.get(i, (i,)):
                observations[j].append(min(clock, end))
            visits.append((min(clock, end), drone.id, i))
            distance += leg
            x, y = point.x, point.y
        distances.append(distance)
        end_drones.append(Drone(drone.id, x, y))
    metrics = gap_metrics(state.points, state.last_seen, observations, end)
    return Evaluation(metrics["objective"], metrics["overdue_count"], metrics["mean_gap_s"],
                      metrics["max_gap_s"], sum(distances), max(distances, default=0.0),
                      metrics["gaps_s"], metrics["observations"], tuple(end_drones), sorted(visits))


def gap_metrics(points, last_seen, observations, end):
    """Longest unobserved interval P_i per point and J = max_i max(0, P_i / q_i - 1).

    Observations are deduplicated and sorted in place. Gaps run from the last
    observation (or activation) through every observation to the window end.
    """
    gaps = {}
    for i in points:
        visits = sorted(set(observations.get(i, [])))
        observations[i] = visits
        boundaries = [last_seen[i], *visits, end]
        gaps[i] = max(b - a for a, b in zip(boundaries, boundaries[1:]))
    overdue = {i: max(0.0, gap / points[i].revisit_s - 1) for i, gap in gaps.items()}
    return {"objective": max(overdue.values(), default=0.0),
            "overdue_count": sum(v > 0 for v in overdue.values()),
            "mean_gap_s": sum(gaps.values()) / max(1, len(gaps)),
            "max_gap_s": max(gaps.values(), default=0.0), "gaps_s": gaps, "observations": observations}


def coverage(points, radius_m):
    ids = sorted(points)
    if not ids:
        return {}
    xy = np.array([(points[i].x, points[i].y) for i in ids])
    near = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1)) <= radius_m + 1e-6
    return {i: tuple(ids[j] for j in np.flatnonzero(row)) for i, row in zip(ids, near)}


class Environment:
    def __init__(self, drones, *, speed_mps=15.0, service_s=5.0, sensor_radius_m=0.0):
        self.drones = tuple(drones)
        if not self.drones or len({d.id for d in self.drones}) != len(self.drones):
            raise ValueError("Drones must have unique IDs and the fleet must be nonempty")
        if not all(math.isfinite(v) for d in self.drones for v in (d.x, d.y)):
            raise ValueError("Drone coordinates must be finite")
        if not all(math.isfinite(v) and v > 0 for v in (speed_mps, service_s)):
            raise ValueError("Speed and service time must be finite and positive")
        if not math.isfinite(sensor_radius_m) or sensor_radius_m < 0:
            raise ValueError("Sensor radius must be finite and non-negative")
        self.speed_mps, self.service_s, self.sensor_radius_m = speed_mps, service_s, sensor_radius_m
        self.time_s = 0.0
        self.points, self.last_seen = {}, {}

    def activate(self, points):
        for i, p in points.items():
            if i != p.id or not all(math.isfinite(v) for v in (p.x, p.y, p.revisit_s)) or p.revisit_s <= 0:
                raise ValueError("Invalid monitoring point")
        self.last_seen = {i: self.last_seen.get(i, self.time_s) for i in points}
        self.points = dict(points)

    def state(self, horizon_s):
        if not math.isfinite(horizon_s) or horizon_s <= 0:
            raise ValueError("Horizon must be finite and positive")
        return PlanningState(self.time_s, horizon_s, self.speed_mps, self.service_s,
                             self.drones, dict(self.points), dict(self.last_seen),
                             coverage(self.points, self.sensor_radius_m) if self.sensor_radius_m else None)

    def mark_observed(self, time_s, ids=None):
        for i in self.points if ids is None else ids:
            if i in self.last_seen:
                self.last_seen[i] = max(self.last_seen[i], time_s)

    def step(self, routes, horizon_s):
        result = evaluate(self.state(horizon_s), routes)
        for i, visits in result.observations.items():
            if visits:
                self.last_seen[i] = visits[-1]
        self.drones = result.end_drones
        self.time_s += horizon_s
        return result


def smoke_router(state: PlanningState):
    """Small deterministic urgency/distance router for plumbing checks only.

    This is not the project's agreed greedy baseline or an ACO implementation.
    Repeated visits are permitted; all scheduled legs fit inside the horizon.
    """
    import heapq

    routes = {d.id: [] for d in state.drones}
    queue = [(state.time_s, d.id, d.x, d.y) for d in state.drones]
    heapq.heapify(queue)
    reserved = dict(state.last_seen)
    end = state.time_s + state.horizon_s
    while queue:
        now, drone_id, x, y = heapq.heappop(queue)
        candidates = []
        for i, p in state.points.items():
            travel = math.hypot(p.x - x, p.y - y) / state.speed_mps + state.service_s
            arrival = now + travel
            if arrival <= end + 1e-9:
                urgency = max(0.0, arrival - reserved[i]) / p.revisit_s
                candidates.append((urgency / travel, i, arrival))
        if not candidates:
            continue
        _, i, arrival = max(candidates)
        p = state.points[i]
        routes[drone_id].append(i)
        reserved[i] = max(reserved[i], arrival)
        heapq.heappush(queue, (arrival, drone_id, p.x, p.y))
    return routes
