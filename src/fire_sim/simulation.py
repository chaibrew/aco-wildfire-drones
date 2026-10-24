"""Deterministic, event-based point-observation simulation."""

import math
from dataclasses import dataclass

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
    distances, end_drones = [], []
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
            observations[i].append(min(clock, end))
            distance += leg
            x, y = point.x, point.y
        distances.append(distance)
        end_drones.append(Drone(drone.id, x, y))
    gaps = {}
    for i in state.points:
        visits = sorted(set(observations[i]))
        observations[i] = visits
        # Preserve overdue age across replanning; new points start at activation.
        boundaries = [state.last_seen[i], *visits, end]
        gaps[i] = max(b - a for a, b in zip(boundaries, boundaries[1:]))
    overdue = {i: max(0.0, gap / state.points[i].revisit_s - 1) for i, gap in gaps.items()}
    return Evaluation(max(overdue.values(), default=0.0),
                      sum(v > 0 for v in overdue.values()),
                      sum(gaps.values()) / max(1, len(gaps)), max(gaps.values(), default=0.0),
                      sum(distances), max(distances, default=0.0), gaps, observations,
                      tuple(end_drones))


class Environment:
    def __init__(self, drones, *, speed_mps=15.0, service_s=5.0):
        self.drones = tuple(drones)
        if not self.drones or len({d.id for d in self.drones}) != len(self.drones):
            raise ValueError("Drones must have unique IDs and the fleet must be nonempty")
        if not all(math.isfinite(v) for d in self.drones for v in (d.x, d.y)):
            raise ValueError("Drone coordinates must be finite")
        if not all(math.isfinite(v) and v > 0 for v in (speed_mps, service_s)):
            raise ValueError("Speed and service time must be finite and positive")
        self.speed_mps, self.service_s = speed_mps, service_s
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
                             self.drones, dict(self.points), dict(self.last_seen))

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
