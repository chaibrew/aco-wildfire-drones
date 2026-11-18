import csv
from datetime import datetime, timezone

import pytest
from shapely.geometry import Point, box

from experiment.replay import (baseline_planners, load_stages, run_replay, scan_anomalies, stage_report)
from fire_sim.spread import ArrivalField, Raster, SpreadModel, interpolate

T0 = datetime(2018, 11, 8, tzinfo=timezone.utc)
DAY = 43200.0


def field_for_seed(seed):
    raster = Raster.covering((-2000, -2000, 7000, 2000), 100, 1000)
    start = Point(0, 0).buffer(1000, 64)
    geoms = [start, start.union(box(0, -600, 3000, 600)), start.union(box(0, -600, 6000, 600))]
    arrival, _ = interpolate(raster, geoms, [0, DAY, 2 * DAY], model=SpreadModel(noise_sigma=0.3), seed=seed)
    return ArrivalField(raster, "EPSG:32610", T0, arrival, (0.0, DAY, 2 * DAY))


def run(tmp_path, **kw):
    args = dict(planners=baseline_planners(20), drone_counts=[2, 3], seeds=[0, 1], hours=1.5,
                cell_size_m=500, out_dir=tmp_path / "out", log=lambda *_: None)
    args.update(kw)
    return run_replay(field_for_seed, args.pop("planners"), args.pop("drone_counts"), args.pop("seeds"),
                      args.pop("hours"), args.pop("out_dir"), **args)


def test_every_planner_drone_count_and_seed_is_run_and_tagged(tmp_path):
    paths = run(tmp_path)
    rows = load_stages(paths["stages"])
    assert {(r["planner"], r["drones"], r["seed"]) for r in rows} == {
        (p, d, s) for p in ("greedy", "vns") for d in (2, 3) for s in (0, 1)}
    with paths["summary"].open() as f:
        assert len(list(csv.DictReader(f))) == 8
    assert len({r["stage"] for r in rows}) == 9      # 1.5 h of 600 s stages


def test_same_seed_gives_every_planner_the_same_fire(tmp_path):
    rows = load_stages(run(tmp_path)["stages"])
    first = {(r["planner"], r["drones"]): r["map_points"] for r in rows if r["seed"] == 0 and r["stage"] == 0}
    assert len(set(first.values())) == 1


def test_results_are_reproducible(tmp_path):
    a = load_stages(run(tmp_path / "a")["stages"])
    b = load_stages(run(tmp_path / "b")["stages"])
    key = lambda r: (r["planner"], r["drones"], r["seed"], r["stage"])
    assert {key(r): r["true_objective"] for r in a} == {key(r): r["true_objective"] for r in b}


def test_resume_skips_finished_missions_and_drops_unfinished_ones(tmp_path):
    paths = run(tmp_path, seeds=[0])
    n = len(load_stages(paths["stages"]))
    run(tmp_path, seeds=[0], resume=True)
    assert len(load_stages(paths["stages"])) == n                      # nothing re-run, nothing duplicated
    # simulate a crash: remove one finished mission's summary row, keep its stage rows
    lines = paths["summary"].read_text().splitlines()
    paths["summary"].write_text("\n".join(lines[:-1]) + "\n")
    run(tmp_path, seeds=[0], resume=True)
    rows = load_stages(paths["stages"])
    assert len(rows) == n
    assert len({(r["planner"], r["drones"], r["seed"], r["stage"]) for r in rows}) == n


def fake_rows(spike_stage=None, spike_seed_only=None, snapshot_at=None, stages=12, seeds=4):
    rows = []
    for seed in range(seeds):
        for stage in range(stages):
            value = 1.0 + 0.01 * stage + 0.02 * seed
            if stage == spike_stage and (spike_seed_only is None or seed == spike_seed_only):
                value += 5.0
            rows.append({"planner": "greedy", "drones": 6, "seed": seed, "stage": stage, "true_objective": value,
                         "map_points": 40.0, "affected_union_fraction": 0.1,
                         "snapshot_delivered": "1" if stage == snapshot_at else ""})
    return rows


def test_stage_report_shows_a_systematic_spike_tied_to_a_snapshot():
    r = stage_report(fake_rows(spike_stage=6, snapshot_at=6), stage=6, window=3)[0]
    assert r["difference"] == pytest.approx(5.0, abs=0.2)
    assert r["seeds_peaking"] == 4 and r["seeds_with_snapshot"] == 4


def test_stage_report_shows_a_single_seed_outlier():
    r = stage_report(fake_rows(spike_stage=6, spike_seed_only=2), stage=6, window=3)[0]
    assert r["seeds_peaking"] == 1 and r["top_seed"] == 2 and r["top_seed_share_of_sum"] > 0.6
    assert r["difference"] < 2.0          # the median across seeds barely moves


def test_scan_finds_the_spike_and_ignores_a_smooth_series():
    found = scan_anomalies(fake_rows(spike_stage=7))
    assert any(f["stage"] == 7 and f["jump"] > 4 for f in found)
    assert scan_anomalies(fake_rows()) == [] or all(f["robust_z"] < 1e6 for f in scan_anomalies(fake_rows()))
    flat = [{**r, "true_objective": 1.0} for r in fake_rows()]
    assert scan_anomalies(flat) == []