import math

import numpy as np
import pytest

from aco_router import ACOParams, ACORouter, StrategyName, diff_instances
from aco_router.aco import _Context
from aco_router.changes import with_neighbourhood
from aco_router.config import ChangeScope
from aco_router.model import Drone, MissionInstance, Point

from experiment import ChangeSpec, ExperimentConfig, apply_change, run_from_states, run_pair, synthetic_state
from experiment.stats import recovery_summary, reset_share_summary

SMALL = ExperimentConfig(n_points=30, drones=3, ants=5, iterations=6, warmup=1)


def line(ids_x, q=300.0):
    return tuple(Point(i, x, 0.0, q, -100.0) for i, x in ids_x)


def instance(points):
    return MissionInstance(points, (Drone("d", 0.0, 0.0),), time=0.0, horizon=600.0)


def test_neighbourhood_covers_points_near_added_removed_and_modified():
    old = instance(line([(0, 0.0), (1, 1000.0), (2, 2000.0), (3, 5000.0), (4, 9000.0)]))
    new = instance(line([(0, 0.0), ("1b", 1000.0 + 0.1), (2, 2000.0), (3, 5000.0), (4, 9000.0)]))
    change = diff_instances(old, new)
    assert change.added == {"1b"} and change.removed == {1} and not change.modified
    near = with_neighbourhood(change, old, new, 1500.0)
    assert near.neighbourhood == {0, "1b", 2}
    assert near.reset_scope == {0, "1b", 2}
    assert with_neighbourhood(change, old, new, 0.0) is change


def _pheromone_after_change(strategy, radius):
    before = synthetic_state(40, 3, seed=3)
    after, info = apply_change(before, ChangeSpec(0.1, "concentrated", "advance"), 1, cell_size_m=1000.0)
    assert info.added and not info.changed
    from aco_router import state_to_instance
    params = ACOParams(n_ants=4, max_iterations=3, selective_radius_m=radius)
    router = ACORouter(params, strategy=strategy, seed=0)
    router.plan(state_to_instance(before))
    router._prepare_pheromone(_Context(state_to_instance(after), params), None)
    return router.pheromone.tau.copy(), router.last_change


def test_without_radius_selective_reset_equals_retention_on_advances():
    sel, _ = _pheromone_after_change(StrategyName.SELECTIVE_RESET, 0.0)
    ret, _ = _pheromone_after_change(StrategyName.RETENTION, 0.0)
    assert np.array_equal(sel, ret)


def test_with_radius_selective_reset_acts_on_advances():
    sel, change = _pheromone_after_change(StrategyName.SELECTIVE_RESET, 2000.0)
    ret, _ = _pheromone_after_change(StrategyName.RETENTION, 2000.0)
    assert not np.array_equal(sel, ret)
    assert change.neighbourhood - change.affected


def test_widespread_trigger_ignores_local_changes():
    base = synthetic_state(40, 3, seed=4)
    from aco_router import state_to_instance
    for fraction, should_act in ((0.1, False), (0.8, True)):
        changed, _ = apply_change(base, ChangeSpec(fraction, "distributed", "risk"), 2, cell_size_m=1000.0)
        params = ACOParams(n_ants=4, max_iterations=3, change_trigger="widespread")
        router = ACORouter(params, strategy=StrategyName.FULL_RESET, seed=0)
        router.plan(state_to_instance(base))
        pre = router.pheromone.copy()
        router._prepare_pheromone(_Context(state_to_instance(changed), params), None)
        assert (router.last_change.scope is ChangeScope.WIDESPREAD) == should_act
        reset = np.all(router.pheromone.tau == router.tau0)
        assert reset == should_act
        if not should_act:
            assert np.array_equal(router.pheromone.tau, pre.remapped(router.pheromone.keys, router.tau0).tau)
            assert router._iters_since_change is None


def test_bad_trigger_and_radius_rejected():
    with pytest.raises(ValueError):
        ACOParams(change_trigger="sometimes")
    with pytest.raises(ValueError):
        ACOParams(selective_radius_m=-1.0)
    with pytest.raises(ValueError):
        ChangeSpec(0.1, direction="sideways")


def test_up_direction_only_raises_risk_and_advances_risk_three():
    base = synthetic_state(80, 4, seed=1)
    new, info = apply_change(base, ChangeSpec(0.4, "distributed", "risk", "up"), 3, cell_size_m=1000.0)
    for i, p in new.points.items():
        if i in base.points and p != base.points[i]:
            assert p.risk == base.points[i].risk + 1
    assert info.added == info.removed > 0
    cyc, _ = apply_change(base, ChangeSpec(0.4, "distributed", "risk"), 3, cell_size_m=1000.0)
    drops = [i for i, p in cyc.points.items() if i in base.points and p.risk < base.points[i].risk]
    assert drops


def test_default_change_is_unchanged_by_the_new_options():
    base = synthetic_state(50, 3, seed=6)
    a, _ = apply_change(base, ChangeSpec(0.3, "concentrated", "mixed"), 9, cell_size_m=1000.0)
    b, _ = apply_change(base, ChangeSpec(0.3, "concentrated", "mixed", "cycle"), 9, cell_size_m=1000.0)
    assert a.points == b.points


def test_planner_filter_and_reset_points_column():
    cfg = ExperimentConfig(n_points=30, drones=3, ants=5, iterations=6, warmup=1, selective_radius_m=1500.0,
                           planners=("aco_retention", "aco_selective_reset"))
    rows, traces = run_pair(synthetic_state(30, 3, seed=0), ChangeSpec(0.2, "concentrated", "advance", "up"), cfg, 0)
    assert [r["planner"] for r in rows] == ["aco_retention", "aco_selective_reset"]
    sel = rows[1]
    assert sel["reset_points"] > sel["added"]
    assert rows[0]["reset_points"] == "" and sel["direction"] == "up"
    vns_only = ExperimentConfig(n_points=30, drones=3, ants=5, iterations=6, planners=("vns",))
    rows, _ = run_pair(synthetic_state(30, 3, seed=0), ChangeSpec(0.2), vns_only, 0)
    assert [r["planner"] for r in rows] == ["vns"]
    with pytest.raises(ValueError):
        ExperimentConfig(planners=("nope",))


def test_parallel_state_runs_match_serial():
    bases = {"a": synthetic_state(25, 3, seed=1), "b": synthetic_state(25, 3, seed=2)}
    cfg = ExperimentConfig(drones=3, ants=4, iterations=4, warmup=1, trials=1, fractions=(0.2,),
                           localities=("concentrated",), planners=("greedy", "aco_retention"))
    serial, _ = run_from_states(bases, cfg, progress=lambda _: None)
    from dataclasses import replace
    parallel, _ = run_from_states(bases, replace(cfg, workers=2), progress=lambda _: None)
    key = lambda r: (r["scenario"], r["planner"], r["total_overdue"], r["margin"])  # noqa: E731
    assert sorted(map(key, serial)) == sorted(map(key, parallel))


def test_recovery_summary_uses_the_incumbent_at_each_checkpoint():
    def tr(planner, trial, pts):
        return [dict(phase="final", scenario="s", trial=trial, target_fraction=0.1, locality="c", planner=planner,
                     evaluations=e, best_total_overdue=v) for e, v in pts]
    rows = []
    for trial in range(3):
        rows += tr("aco_full_reset", trial, [(20, 10.0), (40, 8.0), (60, 6.0)])
        rows += tr("aco_retention", trial, [(20, 7.0), (40, 7.0), (60, 6.0)])
    out = recovery_summary(rows, checkpoints=(20, 40, 60))
    ret = {r["evaluations"]: r for r in out if r["planner"] == "aco_retention"}
    assert math.isclose(ret[20]["mean_diff"], -3.0) and math.isclose(ret[40]["mean_diff"], -1.0)
    assert math.isclose(ret[60]["mean_diff"], 0.0) and ret[20]["better"] == 3


def test_reset_share_bins():
    rows = []
    for trial in range(3):
        for planner, v, reset in (("aco_selective_reset", 5.0, 3), ("aco_retention", 6.0, ""),
                                  ("aco_full_reset", 4.0, "")):
            rows.append(dict(scenario="s", trial=trial, target_fraction=0.1, locality="c", planner=planner,
                             total_overdue=v, reset_points=reset, n_after=30))
    out = reset_share_summary(rows)
    assert {(r["reference"], r["mean_diff"]) for r in out} == {("aco_retention", -1.0), ("aco_full_reset", 1.0)}
    assert all(r["share_lo"] == 0.1 for r in out)
