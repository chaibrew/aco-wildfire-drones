import math

import pytest

from aco_router.changes import diff_instances
from aco_router.config import ChangeScope
from aco_router.evaluator import PlanValidationError, evaluate
from aco_router.kinematics import max_gap, route_schedule
from aco_router.model import Drone, MissionInstance, Plan, Point


def line_instance(q_a=10.0, q_b=40.0, horizon=100.0, last_seen=0.0, coverage=None):
    return MissionInstance(
        points=(Point("A", 100, 0, q_a, last_seen), Point("B", 200, 0, q_b, last_seen)),
        drones=(Drone("d1", 0, 0, speed=10.0),),
        time=0.0,
        horizon=horizon,
        service_s=1.0,
        coverage=coverage,
    )


def test_route_schedule():
    inst = line_instance()
    s = route_schedule(inst, inst.drone("d1"), ["A", "B", "A"])
    assert s.visits == ((11.0, "A"), (22.0, "B"), (33.0, "A"))
    assert s.distance_m == pytest.approx(300.0)
    assert s.end_xy == (100, 0)


def test_evaluate_hand_computed():
    inst = line_instance()
    m = evaluate(inst, Plan({"d1": ["A", "B"]}))
    assert m.gaps_s == {"A": pytest.approx(89.0), "B": pytest.approx(78.0)}
    assert m.objective == pytest.approx(7.9)
    assert m.margin == pytest.approx(8.9)
    assert m.total_overdue == pytest.approx(8.85)
    assert m.overdue_count == 2
    assert m.total_distance_m == pytest.approx(200.0)
    assert m.longest_route_m == pytest.approx(200.0)


def test_objective_equals_clipped_margin_minus_one():
    inst = line_instance(q_a=100.0, q_b=100.0)
    m = evaluate(inst, Plan({"d1": ["A", "B"]}))
    assert m.objective == pytest.approx(max(0.0, m.margin - 1.0))
    assert m.objective == 0.0 and m.total_overdue == 0.0 and m.margin == pytest.approx(0.89)


def test_repeat_visits_split_the_gap():
    m = evaluate(line_instance(), Plan({"d1": ["A", "B", "A"]}))
    assert m.gaps_s["A"] == pytest.approx(67.0)


def test_coverage_observes_neighbours():
    inst = line_instance(coverage={"A": ("A", "B"), "B": ("B",)})
    m = evaluate(inst, Plan({"d1": ["A"]}))
    assert m.gaps_s["B"] == pytest.approx(89.0)


def test_unvisited_point_gap_spans_the_horizon():
    m = evaluate(line_instance(last_seen=-5.0), Plan({"d1": ["A"]}))
    assert m.gaps_s["B"] == pytest.approx(105.0)


def test_last_seen_carries_over():
    fresh = evaluate(line_instance(last_seen=0.0), Plan({"d1": ["A", "B"]}))
    stale = evaluate(line_instance(last_seen=-50.0, horizon=40.0), Plan({"d1": ["A", "B"]}))
    assert stale.gaps_s["A"] == pytest.approx(61.0)
    assert fresh.gaps_s["A"] == pytest.approx(89.0)


def test_new_point_clock_starts_at_stage_time():
    inst = MissionInstance(
        points=(Point("A", 100, 0, 10.0, last_seen=None),),
        drones=(Drone("d1", 0, 0, speed=10.0),),
        time=500.0,
        horizon=100.0,
    )
    assert inst.last_seen(inst.point("A")) == 500.0
    assert max_gap(500.0, [510.0], 600.0) == pytest.approx(90.0)
    assert max_gap(500.0, [510.0, 510.0], 600.0) == pytest.approx(90.0)


@pytest.mark.parametrize(
    "routes, horizon, message",
    [
        ({"d1": ["A", "B"]}, 20.0, "exceeds the horizon"),
        ({"d1": ["A", "Z"]}, 100.0, "unknown point"),
        ({"d1": ["A"], "ghost": []}, 100.0, "unknown drones"),
    ],
)
def test_validation_errors(routes, horizon, message):
    with pytest.raises(PlanValidationError, match=message):
        evaluate(line_instance(horizon=horizon), Plan(routes))


def test_skipping_and_repeating_points_is_allowed():
    evaluate(line_instance(), Plan({"d1": ["B", "A", "B"]}))
    evaluate(line_instance(), Plan({}))


def test_instance_rejects_duplicates_and_future_last_seen():
    with pytest.raises(ValueError, match="duplicate"):
        MissionInstance((Point("A", 0, 0, 1), Point("A", 1, 1, 1)), (Drone("d", 0, 0),))
    with pytest.raises(ValueError, match="after stage time"):
        MissionInstance((Point("A", 0, 0, 1, last_seen=10.0),), (Drone("d", 0, 0),), time=0.0)


def make(points, time=0.0):
    return MissionInstance(points=tuple(points), drones=(Drone("d", 0, 0),), time=time)


def test_diff_by_id_added_removed_modified():
    old = make([Point(1, 0, 0, 10), Point(2, 10, 0, 10), Point(3, 20, 0, 10)])
    new = make([Point(2, 10, 0, 10), Point(3, 25, 0, 10), Point(4, 30, 0, 10)])
    ch = diff_instances(old, new)
    assert ch.added == {4}
    assert ch.removed == {1}
    assert ch.modified == {3}
    assert ch.affected == {3, 4}
    assert ch.fraction == pytest.approx(3 / 4)
    assert ch.scope == ChangeScope.WIDESPREAD


def test_q_change_counts_only_when_enabled():
    old = make([Point(1, 0, 0, 10)])
    new = make([Point(1, 0, 0, 20)])
    assert diff_instances(old, new, count_q_change=True).modified == {1}
    assert diff_instances(old, new, count_q_change=False).is_empty


def test_scope_local_and_none():
    pts = [Point(i, i * 10.0, 0, 10) for i in range(10)]
    old = make(pts)
    same = make(pts)
    assert diff_instances(old, same).scope == ChangeScope.NONE
    one_moved = make(pts[:9] + [Point(9, 999.0, 0, 10)])
    ch = diff_instances(old, one_moved, local_max_fraction=0.2)
    assert ch.scope == ChangeScope.LOCAL and math.isclose(ch.fraction, 0.1)
