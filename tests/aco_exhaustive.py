from __future__ import annotations

import itertools
import math

from aco_router.config import HORIZON_TOLERANCE_S
from aco_router.evaluator import evaluate
from aco_router.model import Drone, Metrics, MissionInstance, Plan

MAX_PLANS = 300_000


def feasible_routes(instance: MissionInstance, drone: Drone) -> list[tuple]:
    limit = instance.end_time + HORIZON_TOLERANCE_S
    out: list[tuple] = [()]

    def extend(route: tuple, x: float, y: float, clock: float) -> None:
        for p in instance.points:
            if route and p.id == route[-1]:
                continue
            nxt = clock + (math.hypot(p.x - x, p.y - y) / drone.speed + instance.service_s)
            if nxt <= limit and nxt > clock:
                r = route + (p.id,)
                out.append(r)
                extend(r, p.x, p.y, nxt)

    extend((), drone.x, drone.y, instance.time)
    return out


def exhaustive_best(instance: MissionInstance, max_plans: int = MAX_PLANS) -> tuple[Plan, Metrics]:
    per_drone = [feasible_routes(instance, d) for d in instance.drones]
    total = math.prod(len(r) for r in per_drone)
    if total > max_plans:
        raise ValueError(f"{total} plans is too many for exhaustive search (max {max_plans})")
    best: tuple[Plan, Metrics] | None = None
    for combo in itertools.product(*per_drone):
        plan = Plan(dict(zip(instance.drone_ids, combo)))
        m = evaluate(instance, plan, validate=False)
        if best is None or m.score < best[1].score:
            best = (plan, m)
    assert best is not None
    return best
