import math
import random

import pytest
from fire_sim import simulation as fs_sim
from fire_sim.models import MonitoringPoint as MP

from aco_router import ACOParams, ACORouter, Plan, StrategyName, evaluate
from aco_router.fire_sim_adapter import FireSimRouter, state_to_instance


def make_env(n=12, drones=2, radius_m=0.0, ring_m=2000.0):
    pts = {}
    for k in range(n):
        a = 2 * math.pi * k / n
        cid = (k, k % 3)
        risk = 1 + k % 3
        pts[cid] = MP(cid, ring_m * math.cos(a), ring_m * math.sin(a), risk, 900 / risk)
    env = fs_sim.Environment([fs_sim.Drone(i, 0.0, 0.0) for i in range(drones)], speed_mps=15, service_s=5,
                             sensor_radius_m=radius_m)
    env.activate(pts)
    return env


def random_routes(state, rng):
    ids = list(state.points)
    routes = {}
    for d in state.drones:
        x, y, clock, route = d.x, d.y, state.time_s, []
        for _ in range(rng.randint(0, 12)):
            i = rng.choice(ids)
            p = state.points[i]
            nxt = clock + math.hypot(p.x - x, p.y - y) / state.speed_mps + state.service_s
            if nxt > state.time_s + state.horizon_s:
                break
            route.append(i)
            x, y, clock = p.x, p.y, nxt
        routes[d.id] = route
    return routes


@pytest.mark.parametrize("radius_m", [0.0, 1500.0])
def test_our_evaluator_matches_fire_sim_exactly(radius_m):
    rng = random.Random(0)
    env = make_env(n=15, drones=3, radius_m=radius_m)
    for stage in range(4):
        state = env.state(600)
        inst = state_to_instance(state)
        for _ in range(25):
            routes = random_routes(state, rng)
            ours = evaluate(inst, Plan(routes))
            theirs = fs_sim.evaluate(state, routes)
            assert ours.objective == theirs.objective
            assert ours.gaps_s == theirs.gaps_s
            assert ours.overdue_count == theirs.overdue_count
            assert ours.total_distance_m == pytest.approx(theirs.total_distance_m)
            assert ours.longest_route_m == pytest.approx(theirs.longest_route_m)
        env.step(random_routes(state, rng), 600)


def test_state_converts_with_q_last_seen_coverage_and_tuple_ids():
    env = make_env(radius_m=1500.0)
    state = env.state(600)
    inst = state_to_instance(state)
    assert set(inst.point_ids) == set(state.points)
    for p in inst.points:
        assert p.q == state.points[p.id].revisit_s
        assert inst.last_seen(p) == state.last_seen[p.id]
        assert set(inst.observed_by(p.id)) == set(state.coverage[p.id])
    assert inst.horizon == 600 and inst.service_s == 5 and all(d.speed == 15 for d in inst.drones)


@pytest.mark.parametrize("strategy", list(StrategyName))
def test_router_runs_several_stages_in_the_environment(strategy):
    env = make_env(radius_m=500.0)
    router = FireSimRouter(ACORouter(ACOParams(n_ants=4, max_iterations=3), strategy=strategy, seed=0))
    for stage in range(3):
        if stage == 2:
            pts = dict(env.points)
            pts.pop(next(iter(pts)))
            k = next(iter(pts))
            pts[k] = MP(k, pts[k].x, pts[k].y, 3, 300.0)
            env.activate(pts)
        state = env.state(600)
        routes = router(state)
        assert set(routes) == {d.id for d in state.drones}
        result = env.step(routes, 600)
        assert result.objective == pytest.approx(router.results[-1].metrics.objective)
    change = router.router.last_change
    assert len(change.removed) == 1 and len(change.modified) >= 1


def test_aco_revisits_an_urgent_point_within_one_stage():
    pts = {(0, 0): MP((0, 0), 300.0, 0.0, 3, 300.0), (9, 9): MP((9, 9), 2000.0, 0.0, 1, 900.0)}
    env = fs_sim.Environment([fs_sim.Drone(0, 0.0, 0.0)], speed_mps=15, service_s=5)
    env.activate(pts)
    state = env.state(600)
    routes = FireSimRouter(ACORouter(ACOParams(n_ants=10, max_iterations=20), seed=0))(state)
    assert routes[0].count((0, 0)) >= 2
    assert fs_sim.evaluate(state, routes).objective == 0.0
