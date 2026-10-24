import pytest

from fire_sim.models import MonitoringPoint
from fire_sim.simulation import Drone, Environment, evaluate, smoke_router


def point(i=(0, 0), x=0, y=0, q=5):
    return MonitoringPoint(i, x, y, 1, q)


def test_gaps_include_window_edges_and_unvisited_points():
    env = Environment([Drone(0, 0, 0)], speed_mps=1, service_s=1)
    env.activate({(0, 0): point(x=3), (1, 0): point((1, 0), x=20, q=2)})
    result = evaluate(env.state(10), {0: [(0, 0)]})
    assert result.observations[(0, 0)] == [4]
    assert result.gaps_s == {(0, 0): 6, (1, 0): 10}
    assert result.objective == 4
    assert result.overdue_count == 2
    assert result.total_distance_m == 3
    assert env.time_s == 0  # Evaluation is pure.


def test_observation_history_and_position_persist_across_replanning():
    env = Environment([Drone(0, 0, 0)], speed_mps=1, service_s=1)
    points = {(0, 0): point(x=3)}
    env.activate(points)
    env.step({0: [(0, 0)]}, 10)
    env.activate(points)
    result = env.step({0: []}, 10)
    assert result.gaps_s[(0, 0)] == 16
    assert env.drones[0].x == 3
    env.activate({})
    env.step({0: []}, 10)
    env.activate(points)
    assert env.last_seen[(0, 0)] == 30  # Reappearance starts new active lifetime.


def test_repeated_and_simultaneous_visits_and_exact_deadline():
    env = Environment([Drone(0, 0, 0), Drone(1, 0, 0)], service_s=2)
    env.activate({(0, 0): point(q=2)})
    result = env.step({0: [(0, 0)] * 3, 1: [(0, 0)]}, 6)
    assert result.observations[(0, 0)] == [2, 4, 6]
    assert result.objective == 0


@pytest.mark.parametrize("routes", [{}, {0: [(99, 99)]}, {0: [(0, 0)] * 20}])
def test_invalid_routes_do_not_mutate_state(routes):
    env = Environment([Drone(0, 0, 0)], service_s=1)
    env.activate({(0, 0): point()})
    before = env.state(10)
    with pytest.raises(ValueError):
        env.step(routes, 10)
    assert env.state(10) == before


def test_empty_active_set_and_unreachable_nodes():
    env = Environment([Drone(0, 0, 0)], service_s=1)
    assert env.step({0: []}, 10).objective == 0
    env.activate({(0, 0): point(x=1e6)})
    assert smoke_router(env.state(10)) == {0: []}


def test_smoke_routes_are_feasible_and_deterministic():
    env = Environment([Drone(i, 0, 0) for i in range(3)])
    env.activate({(i, 0): point((i, 0), x=100 * i) for i in range(8)})
    state = env.state(100)
    assert smoke_router(state) == smoke_router(state)
    result = evaluate(state, smoke_router(state))
    assert result.total_distance_m > 0
    assert all(t <= 100 for visits in result.observations.values() for t in visits)
