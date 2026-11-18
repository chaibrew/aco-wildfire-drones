from __future__ import annotations

from typing import Any, Mapping

from fire_sim.simulation import evaluate as fire_sim_evaluate

from .aco import ACORouter
from .changes import diff_instances
from .evaluator import make_metrics
from .model import Drone, Metrics, MissionInstance, Plan, PlanResult, Point


def state_to_instance(state: Any, stage: int = 0) -> MissionInstance:
    points = tuple(
        Point(pid, p.x, p.y, p.revisit_s, state.last_seen.get(pid)) for pid, p in sorted(state.points.items())
    )
    drones = tuple(Drone(d.id, d.x, d.y, state.speed_mps) for d in state.drones)
    coverage = None if state.coverage is None else {k: tuple(v) for k, v in state.coverage.items()}
    return MissionInstance(points, drones, time=state.time_s, horizon=state.horizon_s, stage=stage,
                           service_s=state.service_s, coverage=coverage)


def metrics_from_evaluation(state: Any, ev: Any) -> Metrics:
    q = {i: p.revisit_s for i, p in state.points.items()}
    return make_metrics(q, ev.gaps_s, ev.total_distance_m, ev.longest_route_m)


def fire_sim_evaluator(state: Any):
    drone_ids = [d.id for d in state.drones]

    def score(instance: MissionInstance, plan: Plan) -> Metrics:
        return metrics_from_evaluation(state, fire_sim_evaluate(state, plan.for_drones(drone_ids)))

    return score


class FireSimRouter:
    def __init__(self, router: ACORouter):
        self.router = router
        self.stage = 0
        self.results: list[PlanResult] = []
        self._previous: MissionInstance | None = None

    @property
    def name(self) -> str:
        return self.router.name

    def __call__(self, state: Any) -> dict[Any, list]:
        instance = state_to_instance(state, self.stage)
        change = None if self._previous is None else diff_instances(self._previous, instance)
        self.router.evaluator = fire_sim_evaluator(state)
        result = self.router.plan(instance, change)
        self.results.append(result)
        self._previous = instance
        self.stage += 1
        return result.plan.for_drones(instance.drone_ids)


def _mean(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [r[key] for r in rows if key in r and r[key] == r[key]]
    return sum(values) / len(values) if values else float("nan")


def summary(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    n = max(1, len(rows))
    return {
        "stages": len(rows),
        "map_J_mean": sum(r["map_objective"] for r in rows) / n,
        "map_J_max": max((r["map_objective"] for r in rows), default=0.0),
        "map_overdue_mean": sum(r["map_overdue_count"] for r in rows) / n,
        "true_J_mean": sum(r["true_objective"] for r in rows) / n,
        "true_overdue_mean": sum(r["true_overdue_count"] for r in rows) / n,
        "front_on_map_mean": sum(r["front_on_map_fraction"] for r in rows) / n,
        "perimeter_error_mean_m": _mean(rows, "perimeter_error_m"),
        "perimeter_error_p95_mean_m": _mean(rows, "perimeter_error_p95_m"),
        "growth_found_mean": _mean(rows, "growth_found_fraction"),
        "known_burned_mean": _mean(rows, "known_burned_fraction"),
        "planning_s_mean": sum(r["planning_s"] for r in rows) / n,
    }
