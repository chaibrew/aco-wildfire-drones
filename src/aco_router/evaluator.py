from __future__ import annotations

from typing import Mapping

from .config import HORIZON_TOLERANCE_S
from .kinematics import max_gap, route_schedule
from .model import Metrics, MissionInstance, Plan, PointId


class PlanValidationError(ValueError):
    pass


def validate_plan(instance: MissionInstance, plan: Plan) -> None:
    unknown_drones = set(plan.routes) - set(instance.drone_ids)
    if unknown_drones:
        raise PlanValidationError(f"unknown drones: {sorted(map(repr, unknown_drones))}")
    points = instance.points_by_id
    end = instance.end_time
    for drone in instance.drones:
        route = plan.route(drone.id)
        for pid in route:
            if pid not in points:
                raise PlanValidationError(f"drone {drone.id!r} visits unknown point {pid!r}")
        sched = route_schedule(instance, drone, route)
        if sched.end_time > end + HORIZON_TOLERANCE_S:
            raise PlanValidationError(f"drone {drone.id!r} route exceeds the horizon")


def evaluate(instance: MissionInstance, plan: Plan, *, validate: bool = True) -> Metrics:
    if validate:
        validate_plan(instance, plan)
    end = instance.end_time
    observations: dict[PointId, list[float]] = {p.id: [] for p in instance.points}
    distances = []
    for drone in instance.drones:
        sched = route_schedule(instance, drone, plan.route(drone.id))
        distances.append(sched.distance_m)
        for t, pid in sched.visits:
            for seen in instance.observed_by(pid):
                if seen in observations:
                    observations[seen].append(min(t, end))
    gaps = {p.id: max_gap(instance.last_seen(p), observations[p.id], end) for p in instance.points}
    q = {p.id: p.q for p in instance.points}
    return make_metrics(q, gaps, sum(distances), max(distances, default=0.0))


def make_metrics(q: Mapping[PointId, float], gaps: Mapping[PointId, float], total_distance_m: float,
                 longest_route_m: float) -> Metrics:
    if not gaps:
        return Metrics(0.0, 0.0, 0.0, 0, 0.0, 0.0, total_distance_m, longest_route_m, {})
    ratios = {i: g / q[i] for i, g in gaps.items()}
    overdue = {i: max(0.0, r - 1.0) for i, r in ratios.items()}
    values = list(gaps.values())
    return Metrics(
        objective=max(overdue.values()),
        margin=max(ratios.values()),
        total_overdue=sum(overdue.values()),
        overdue_count=sum(v > 0 for v in overdue.values()),
        mean_gap_s=sum(values) / len(values),
        max_gap_s=max(values),
        total_distance_m=total_distance_m,
        longest_route_m=longest_route_m,
        gaps_s=dict(gaps),
    )
