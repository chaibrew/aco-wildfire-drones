from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .model import Drone, MissionInstance, PointId


@dataclass(frozen=True)
class RouteSchedule:
    visits: tuple[tuple[float, PointId], ...]
    distance_m: float
    end_xy: tuple[float, float]
    end_time: float


def route_schedule(instance: MissionInstance, drone: Drone, route: Sequence[PointId]) -> RouteSchedule:
    x, y, clock, distance = drone.x, drone.y, instance.time, 0.0
    visits = []
    for pid in route:
        p = instance.point(pid)
        leg = math.hypot(p.x - x, p.y - y)
        clock += leg / drone.speed + instance.service_s
        distance += leg
        visits.append((clock, pid))
        x, y = p.x, p.y
    return RouteSchedule(tuple(visits), distance, (x, y), clock)


def max_gap(last_seen: float, observation_times: Iterable[float], t_end: float) -> float:
    boundaries = [last_seen, *sorted(set(observation_times)), t_end]
    return max(b - a for a, b in zip(boundaries, boundaries[1:]))
