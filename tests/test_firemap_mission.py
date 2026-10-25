from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import Point, box

from fire_sim.firemap import FireMap
from fire_sim.mission import MissionConfig, run_mission
from fire_sim.models import MonitoringPoint
from fire_sim.simulation import Drone, Environment, evaluate
from fire_sim.spread import ArrivalField, Raster, SpreadModel, interpolate

T0 = datetime(2018, 11, 8, tzinfo=timezone.utc)
DAY = 43200.0


@pytest.fixture(scope="module")
def field():
    raster = Raster.covering((-2000, -2000, 7000, 2000), 100, 1000)
    start = Point(0, 0).buffer(1000, 64)
    geoms = [start, start.union(box(0, -600, 3000, 600)), start.union(box(0, -600, 6000, 600))]
    arrival, _ = interpolate(raster, geoms, [0, DAY, 2 * DAY], model=SpreadModel(noise_sigma=0))
    return ArrivalField(raster, "EPSG:32610", T0, arrival, (0.0, DAY, 2 * DAY))


def test_snapshot_sets_map_and_evidence(field):
    fire_map = FireMap(field, cell_size_m=500, sensor_radius_m=500)
    fire_map.ingest_snapshot(0)
    assert np.array_equal(fire_map.known, field.arrival_s <= 0)
    assert all(p.risk == 1 for p in fire_map.points().values())
    fire_map.ingest_snapshot(1)
    assert np.array_equal(fire_map.known, field.arrival_s <= DAY)
    points = fire_map.points()
    head = max(points.values(), key=lambda p: p.x)
    assert fire_map.evidence[head.id][0] > 0
    assert all(fire_map.evidence[p.id][0] == 0 for p in points.values() if p.x < -500)


def test_drone_sees_only_inside_its_footprint(field):
    fire_map = FireMap(field, cell_size_m=500, sensor_radius_m=500)
    fire_map.ingest_snapshot(1)
    t = 1.5 * DAY
    before = fire_map.known.copy()
    fire_map.observe(3250, 0, t)
    x, y = field.raster.centres()
    inside = np.hypot(x - 3250, y) <= 500
    truth = field.arrival_s <= t
    assert np.array_equal(fire_map.known[inside], (before | truth)[inside])
    assert np.array_equal(fire_map.known[~inside], before[~inside])
    assert (fire_map.known & ~before).any()
    assert fire_map.evidence[(6, 0)][0] > 0


def test_looking_without_seeing_advance_records_zero(field):
    fire_map = FireMap(field, cell_size_m=500, sensor_radius_m=500)
    fire_map.ingest_snapshot(1)
    fire_map.observe(-1000, 0, 1.5 * DAY)
    looked = [value for value, when in fire_map.evidence.values() if when == 1.5 * DAY]
    assert looked and all(value == 0 for value in looked)


def test_footprint_observes_nearby_points():
    points = {(0, 0): MonitoringPoint((0, 0), 0, 0, 1, 100), (1, 0): MonitoringPoint((1, 0), 400, 0, 1, 100),
              (5, 0): MonitoringPoint((5, 0), 2000, 0, 1, 100)}
    env = Environment([Drone(0, 0, 0)], speed_mps=10, service_s=1, sensor_radius_m=500)
    env.activate(points)
    result = evaluate(env.state(10), {0: [(0, 0)]})
    assert result.observations[(0, 0)] == [1] and result.observations[(1, 0)] == [1]
    assert result.observations[(5, 0)] == []
    assert result.visits == [(1, 0, (0, 0))]


def test_planner_never_sees_truth_between_snapshots(field):
    seen = []

    def idle(state):
        seen.append((state.time_s, set(state.points)))
        return {d.id: [] for d in state.drones}

    config = MissionConfig(cell_size_m=500, horizon_s=3600, drones=1)
    rows, blocks = run_mission(field, idle, config)
    snapshot0 = FireMap(field, cell_size_m=500)
    snapshot0.ingest_snapshot(0)
    snapshot1 = FireMap(field, cell_size_m=500)
    snapshot1.ingest_snapshot(1)
    for t, ids in seen:
        assert ids == set((snapshot0 if t < DAY else snapshot1).points())
    lagging = [r["front_on_map_fraction"] for r in rows if 6 * 3600 <= r["time_s"] < DAY]
    assert min(lagging) < 0.5
    assert [r["snapshot_delivered"] for r in rows if r["snapshot_delivered"]] == ["1"]
    assert [b["block"] for b in blocks] == [0, 1]
    assert [b["end_snapshot"] for b in blocks] == [1, 2]
    assert all(b["discovered_growth_fraction"] == 0 for b in blocks)
    assert all(b["perimeter_error_m"] > 500 for b in blocks)
    assert blocks[0]["snapshot_changed_fraction"] is None and blocks[1]["snapshot_changed_fraction"] > 0


def test_snapshot_score_with_full_knowledge(field):
    fire_map = FireMap(field, cell_size_m=500)
    fire_map.ingest_snapshot(0)
    fire_map.known = field.arrival_s <= DAY
    score = fire_map.score_snapshot(1)
    assert score["discovered_growth_fraction"] == 1.0 and score["perimeter_error_m"] == 0.0
    with pytest.raises(ValueError):
        fire_map.score_snapshot(0)


def test_flying_fleet_discovers_growth(field):
    from fire_sim.simulation import smoke_router
    _, blocks = run_mission(field, smoke_router, MissionConfig(cell_size_m=500, drones=3), stop_s=DAY)
    assert len(blocks) == 1 and blocks[0]["end_snapshot"] == 1
    assert blocks[0]["discovered_growth_fraction"] > 0.3


def test_mission_rows_and_truth_scoring(field):
    from fire_sim.simulation import smoke_router
    rows, _ = run_mission(field, smoke_router, MissionConfig(cell_size_m=500, drones=2), stop_s=6 * 3600)
    assert len(rows) == 36
    assert all(r["visits"] > 0 for r in rows)
    assert all(0 <= r["front_on_map_fraction"] <= 1 for r in rows)
    with pytest.raises(ValueError):
        run_mission(field, smoke_router, MissionConfig(), start_s=3 * DAY)
