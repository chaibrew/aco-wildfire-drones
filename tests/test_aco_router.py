import math

import pytest

from aco_router.aco import ACORouter
from aco_router.config import ACOParams, BudgetKind, Tau0Mode
from aco_router.evaluator import validate_plan
from aco_router.model import Drone, MissionInstance, Point

from aco_exhaustive import exhaustive_best
from aco_helpers import SMALL_PARAMS, random_instance, tiny_instance


def next_visit_fits(inst, drone, route):
    x, y, clock = drone.x, drone.y, inst.time
    for pid in route:
        p = inst.point(pid)
        clock += math.hypot(p.x - x, p.y - y) / drone.speed + inst.service_s
        x, y = p.x, p.y
    return any(
        clock + math.hypot(p.x - x, p.y - y) / drone.speed + inst.service_s <= inst.end_time
        for p in inst.points
        if not route or p.id != route[-1]
    )


def test_routes_fit_the_horizon_and_drones_keep_flying():
    inst = random_instance(1, n_points=25, n_drones=4)
    res = ACORouter(SMALL_PARAMS, seed=0).plan(inst)
    validate_plan(inst, res.plan)
    for d in inst.drones:
        assert res.plan.route(d.id), "every drone should fly"
        assert not next_visit_fits(inst, d, res.plan.route(d.id))


def test_same_seed_same_plan():
    inst = random_instance(2, n_points=15, n_drones=3)
    a = ACORouter(SMALL_PARAMS, seed=7).plan(inst)
    b = ACORouter(SMALL_PARAMS, seed=7).plan(inst)
    assert a.plan == b.plan
    assert a.metrics.score == b.metrics.score


def test_different_seeds_can_differ():
    inst = random_instance(3, n_points=20, n_drones=3)
    plans = {repr(ACORouter(SMALL_PARAMS.with_(max_iterations=2), seed=s).plan(inst).plan) for s in range(5)}
    assert len(plans) > 1


def test_matches_exhaustive_search_on_tiny_instances():
    hits = 0
    for seed in range(15):
        inst = tiny_instance(seed)
        _, best = exhaustive_best(inst)
        m = ACORouter(ACOParams(n_ants=20, max_iterations=60), seed=seed).plan(inst).metrics
        hits += m.margin <= best.margin + 1e-9 and m.total_overdue <= best.total_overdue + 1e-9
    assert hits >= 14


def test_tie_break_still_steers_when_one_point_is_out_of_reach():
    base = tiny_instance(3)
    far = Point("far", 50_000.0, 0.0, 30.0, last_seen=-100.0)
    inst = MissionInstance(base.points + (far,), base.drones, time=0.0, horizon=base.horizon, service_s=5.0)
    _, best = exhaustive_best(inst)
    m = ACORouter(ACOParams(n_ants=20, max_iterations=60), seed=0).plan(inst).metrics
    assert m.margin == pytest.approx(best.margin)
    assert m.total_overdue == pytest.approx(best.total_overdue)


def test_trace_is_monotone_and_ends_at_result():
    inst = random_instance(4, n_points=20, n_drones=3)
    res = ACORouter(SMALL_PARAMS, seed=1).plan(inst)
    assert len(res.trace) == res.iterations == SMALL_PARAMS.max_iterations
    margins = [t.best_margin for t in res.trace]
    assert all(b <= a + 1e-12 for a, b in zip(margins, margins[1:]))
    assert res.trace[-1].best_margin == pytest.approx(res.metrics.margin)
    assert res.trace[-1].best_total_overdue == pytest.approx(res.metrics.total_overdue)


def test_time_budget_is_respected():
    inst = random_instance(5, n_points=30, n_drones=4)
    params = ACOParams(n_ants=5, budget_kind=BudgetKind.SECONDS, time_budget_s=0.3)
    res = ACORouter(params, seed=0).plan(inst)
    assert res.iterations >= 1
    assert res.runtime_s < 0.3 + 0.5


def test_learning_does_not_lose_to_one_iteration():
    inst = random_instance(6, n_points=30, n_drones=3)
    one = ACORouter(ACOParams(n_ants=10, max_iterations=1), seed=0).plan(inst)
    many = ACORouter(ACOParams(n_ants=10, max_iterations=60), seed=0).plan(inst)
    assert many.metrics.score <= one.metrics.score


def test_tau0_modes():
    inst = random_instance(7, n_points=10, n_drones=2)
    const = ACORouter(ACOParams(tau0_mode=Tau0Mode.CONSTANT, tau0_constant=3.0, max_iterations=1), seed=0)
    const.plan(inst)
    assert const.tau0 == 3.0
    nn = ACORouter(ACOParams(max_iterations=1), seed=0)
    nn.plan(inst)
    assert nn.tau0 > 0 and nn.tau0 != 3.0


def test_no_points_and_unreachable_points():
    empty = MissionInstance((), (Drone("a", 0, 0),))
    assert ACORouter(SMALL_PARAMS).plan(empty).plan.to_dict() == {"a": []}
    far = MissionInstance((Point(1, 1e6, 0, 300.0),), (Drone("a", 0, 0),), horizon=600.0)
    res = ACORouter(SMALL_PARAMS).plan(far)
    assert res.plan.route("a") == () and res.metrics.overdue_count == 1
