import numpy as np
import pytest

from aco_router.aco import ACORouter
from aco_router.changes import diff_instances
from aco_router.config import ACOParams, ChangeScope, StrategyName
from aco_router.evaluator import validate_plan
from aco_router.model import ChangeEvent, MissionInstance, Point
from aco_router.pheromone import DroneNode, PheromoneMatrix
from aco_router.strategies import STRATEGIES, FastEvaporation, make_strategy

from aco_helpers import random_instance

TAU0 = 1.0


def matrix_with_values():
    keys = (DroneNode("d"), "a", "b", "c")
    m = PheromoneMatrix(keys, TAU0)
    m.tau[:] = np.arange(16, dtype=float).reshape(4, 4) + 10.0
    return m


def test_remap_keeps_surviving_edges_and_initialises_new_ones():
    old = matrix_with_values()
    new = old.remapped((DroneNode("d"), "b", "c", "z"), TAU0)
    assert new.get("b", "c") == old.get("b", "c")
    assert new.get(DroneNode("d"), "c") == old.get(DroneNode("d"), "c")
    assert new.get("b", "z") == TAU0 and new.get("z", "c") == TAU0
    assert "a" not in new.index


def test_retention_keeps_everything():
    m = matrix_with_values()
    before = m.tau.copy()
    make_strategy(StrategyName.RETENTION).on_change(m, ChangeEvent(added=frozenset({"a"})), TAU0)
    assert np.array_equal(m.tau, before)


def test_full_reset_sets_every_edge_to_tau0():
    m = matrix_with_values()
    make_strategy(StrategyName.FULL_RESET).on_change(m, ChangeEvent(modified=frozenset({"a"})), TAU0)
    assert np.all(m.tau == TAU0)


def test_selective_reset_touches_only_edges_incident_to_ct():
    m = matrix_with_values()
    before = m.tau.copy()
    change = ChangeEvent(modified=frozenset({"b"}))
    make_strategy(StrategyName.SELECTIVE_RESET).on_change(m, change, TAU0)
    b = m.index["b"]
    for i in range(4):
        for j in range(4):
            if i == b or j == b:
                assert m.tau[i, j] == TAU0
            else:
                assert m.tau[i, j] == before[i, j]


def test_fast_evaporation_schedule():
    params = ACOParams(rho=0.1, rho_high=0.6, fast_evaporation_iterations=3)
    s = FastEvaporation()
    assert [s.evaporation_rate(params, k) for k in (0, 1, 2, 3, 10)] == [0.6, 0.6, 0.6, 0.1, 0.1]
    assert s.evaporation_rate(params, None) == 0.1
    for other in (StrategyName.RETENTION, StrategyName.SELECTIVE_RESET, StrategyName.FULL_RESET):
        assert make_strategy(other).evaporation_rate(params, 0) == 0.1


def test_registry_covers_every_strategy_name():
    assert set(STRATEGIES) == set(StrategyName)
    for name in StrategyName:
        assert make_strategy(name.value).name is name


def shifted(inst: MissionInstance, moved_ids, dx=400.0, time_step=900.0) -> MissionInstance:
    pts = tuple(
        Point(p.id, p.x + (dx if p.id in moved_ids else 0.0), p.y, p.q) for p in inst.points
    )
    return MissionInstance(pts, inst.drones, time=inst.time + time_step, horizon=inst.horizon, stage=inst.stage + 1)


@pytest.mark.parametrize("strategy", list(StrategyName))
def test_router_applies_strategy_on_auto_detected_change(strategy):
    params = ACOParams(n_ants=8, max_iterations=10)
    before = random_instance(11, n_points=12, n_drones=2)
    after = shifted(before, moved_ids={0, 1})
    router = ACORouter(params, strategy=strategy, seed=0)
    router.plan(before)
    pre = router.pheromone.copy()
    tau0 = router.tau0

    from aco_router.aco import _Context

    router._prepare_pheromone(_Context(after, params), None)
    ch = router.last_change
    assert ch.modified == {0, 1} and ch.scope is ChangeScope.LOCAL
    post = router.pheromone
    untouched = [k for k in post.keys if k not in (0, 1)]
    i, j = untouched[2], untouched[3]
    if strategy is StrategyName.FULL_RESET:
        assert np.all(post.tau == tau0)
    elif strategy is StrategyName.SELECTIVE_RESET:
        assert post.get(0, i) == tau0 and post.get(i, 1) == tau0
        assert post.get(i, j) == pre.get(i, j)
    else:
        assert post.get(0, i) == pre.get(0, i)
        assert post.get(i, j) == pre.get(i, j)


def test_router_plans_across_point_additions_and_removals():
    params = ACOParams(n_ants=8, max_iterations=10)
    before = random_instance(12, n_points=10, n_drones=2)
    pts = tuple(p for p in before.points if p.id != 0) + (Point(99, 20000.0, 20000.0, 300.0),)
    after = MissionInstance(pts, before.drones, time=900.0, horizon=before.horizon, stage=1)
    router = ACORouter(params, strategy=StrategyName.SELECTIVE_RESET, seed=0)
    router.plan(before)
    res = router.plan(after)
    assert router.last_change.added == {99} and router.last_change.removed == {0}
    assert 0 not in router.pheromone.index and 99 in router.pheromone.index
    validate_plan(after, res.plan)
    assert all(p in after.points_by_id for r in res.plan.routes.values() for p in r)


def test_explicit_change_overrides_auto_diff():
    params = ACOParams(n_ants=4, max_iterations=2)
    before = random_instance(13, n_points=8, n_drones=2)
    after = shifted(before, moved_ids={0})
    router = ACORouter(params, strategy=StrategyName.SELECTIVE_RESET)
    router.plan(before)
    forced = diff_instances(before, after)
    router.plan(after, forced)
    assert router.last_change is forced


def test_unchanged_instance_leaves_fast_evaporation_off():
    params = ACOParams(n_ants=4, max_iterations=3)
    inst = random_instance(14, n_points=8, n_drones=2)
    router = ACORouter(params, strategy=StrategyName.FAST_EVAPORATION)
    router.plan(inst)
    router.plan(inst)
    assert router.last_change.is_empty
    assert router._iters_since_change is None
