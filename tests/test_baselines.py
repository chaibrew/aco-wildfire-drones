import math
import random

from fire_sim import simulation as fs
from fire_sim.models import MonitoringPoint as MP

from aco_router import state_to_instance
from baselines import GreedyRouter, VNSParams, VNSRouter

from aco_exhaustive import exhaustive_best


def ring_state(n=20, drones=3, radius_m=0.0, ring_m=6000.0, seed=0, horizon=600.0):
    rng = random.Random(seed)
    pts = {}
    for k in range(n):
        a = 2 * math.pi * k / n
        cid = (k, 0)
        risk = rng.choice((1, 2, 3))
        pts[cid] = MP(cid, ring_m * math.cos(a), ring_m * math.sin(a), risk, 900 / risk)
    ds = [fs.Drone(d, pts[(d * n // drones, 0)].x, pts[(d * n // drones, 0)].y) for d in range(drones)]
    env = fs.Environment(ds, speed_mps=15, service_s=5, sensor_radius_m=radius_m)
    env.activate(pts)
    return env, env.state(horizon)


def tiny_state(seed):
    rng = random.Random(seed)
    pts = {}
    for k in range(4):
        q = rng.choice((60.0, 120.0, 240.0))
        pts[(k, 0)] = MP((k, 0), rng.uniform(-1200, 1200), rng.uniform(-1200, 1200), 1, q)
    env = fs.Environment([fs.Drone(0, 0.0, 0.0), fs.Drone(1, 0.0, 0.0)], speed_mps=15, service_s=5)
    env.activate(pts)
    env.last_seen = {i: -rng.uniform(0, 200) for i in pts}
    return env.state(220.0)


def test_greedy_is_feasible_deterministic_and_covers_every_drone_key():
    for radius in (0.0, 7000.0):
        _, st = ring_state(radius_m=radius)
        a = GreedyRouter()(st)
        assert a == GreedyRouter()(st)
        assert set(a) == {d.id for d in st.drones}
        fs.evaluate(st, a)  # raises if a route exceeds the horizon or names an inactive point


def test_greedy_routes_are_maximal_nothing_else_fits():
    _, st = ring_state()
    routes = GreedyRouter()(st)
    for d in st.drones:
        x, y, clock = d.x, d.y, st.time_s
        for i in routes[d.id]:
            p = st.points[i]
            clock += math.hypot(p.x - x, p.y - y) / st.speed_mps + st.service_s
            x, y = p.x, p.y
        assert not any(clock + math.hypot(p.x - x, p.y - y) / st.speed_mps + st.service_s <= st.time_s + st.horizon_s
                       for p in st.points.values())


def test_greedy_handles_no_points():
    env = fs.Environment([fs.Drone(0, 0.0, 0.0)])
    assert GreedyRouter()(env.state(600.0)) == {0: []}


def test_vns_feasible_budgeted_and_never_worse_than_its_greedy_start():
    for seed in range(5):
        _, st = ring_state(n=25, drones=4, seed=seed)
        inst = state_to_instance(st)
        vns = VNSRouter(VNSParams(max_evaluations=120), seed=seed)
        routes = vns(st)
        res = vns.results[0]
        fs.evaluate(st, routes)
        assert res.evaluations <= 120
        from aco_router import Plan, evaluate

        greedy_m = evaluate(inst, Plan(GreedyRouter()(st)))
        assert res.metrics.score <= greedy_m.score
        margins = [t.best_margin for t in res.trace]
        assert all(b <= a + 1e-12 for a, b in zip(margins, margins[1:]))
        assert res.trace[-1].iteration == res.evaluations


def test_vns_same_seed_same_routes_and_evaluations_count_distinct_plans():
    _, st = ring_state(n=25, drones=4)
    a, b = VNSRouter(VNSParams(max_evaluations=80), seed=3), VNSRouter(VNSParams(max_evaluations=80), seed=3)
    assert a(st) == b(st)
    assert a.results[0].evaluations == b.results[0].evaluations


def test_vns_handles_empty_and_single_point_states():
    env = fs.Environment([fs.Drone(0, 0.0, 0.0)])
    assert VNSRouter(seed=0)(env.state(600.0)) == {0: []}
    env.activate({(0, 0): MP((0, 0), 100.0, 0.0, 1, 900.0)})
    routes = VNSRouter(VNSParams(max_evaluations=20), seed=0)(env.state(600.0))
    fs.evaluate(env.state(600.0), routes)


def test_vns_is_near_optimal_on_tiny_instances():
    """Equal budget to the ACO exhaustive test (1200 evaluations).

    VNS must match the exhaustive optimum on the primary criterion (margin) every time and stay
    within a few percent on the secondary one (total overdue). It is a heuristic: it does not
    always find the exact optimum (about 11 of 15 here), and the report should say so.
    """
    gaps = []
    for seed in range(15):
        st = tiny_state(seed)
        _, best = exhaustive_best(state_to_instance(st))
        vns = VNSRouter(VNSParams(max_evaluations=1200), seed=seed)
        vns(st)
        m = vns.results[0].metrics
        assert m.margin <= best.margin + 1e-9, (seed, m.margin, best.margin)
        gaps.append((m.total_overdue - best.total_overdue) / best.total_overdue)
    assert min(gaps) >= -1e-9 and sum(gaps) / len(gaps) < 0.05, gaps


def test_vns_stops_when_the_search_space_is_exhausted_instead_of_spinning():
    st = tiny_state(0)
    vns = VNSRouter(VNSParams(max_evaluations=10**7, patience=30), seed=0)
    vns(st)  # must return; the space has ~3e4 plans, far fewer than 1e7 evaluations
    assert vns.results[0].evaluations < 10**7


def test_baselines_run_inside_the_environment_loop_for_many_stages():
    env, _ = ring_state(n=30, drones=3, radius_m=1500.0)
    vns = VNSRouter(VNSParams(max_evaluations=40), seed=1)
    greedy = GreedyRouter()
    for stage in range(6):
        st = env.state(600.0)
        env.step((vns if stage % 2 else greedy)(st), 600.0)  # step() raises on any rule violation
