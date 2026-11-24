import math

from aco_router import StrategyName

from experiment import ChangeSpec, ExperimentConfig, apply_change, run_pair, synthetic_state
from experiment.perturb import KINDS, LOCALITIES
from experiment.runner import PLANNERS
from experiment.stats import load_rows, paired_summary, trace_summary

SMALL = ExperimentConfig(n_points=30, drones=3, ants=5, iterations=6, warmup=1)


def test_synthetic_state_is_valid_and_reproducible():
    a, b = synthetic_state(40, 4, seed=3), synthetic_state(40, 4, seed=3)
    assert a.points == b.points and a.last_seen == b.last_seen
    assert len(a.points) == 40 and len(a.drones) == 4
    assert set(a.last_seen) == set(a.points) and all(t <= a.time_s for t in a.last_seen.values())
    assert set(a.coverage) == set(a.points)
    assert synthetic_state(40, 4, seed=4).points != a.points


def test_realised_fraction_tracks_target_for_every_kind():
    base = synthetic_state(80, 4, seed=1)
    for kind in KINDS:
        for loc in LOCALITIES:
            for f in (0.05, 0.2, 0.5):
                _, info = apply_change(base, ChangeSpec(f, loc, kind), 0, cell_size_m=1000.0)
                assert abs(info.realised_fraction - f) < 0.03, (kind, loc, f, info)


def test_zero_fraction_changes_nothing():
    base = synthetic_state(30, 3, seed=2)
    new, info = apply_change(base, ChangeSpec(0.0), 0, cell_size_m=1000.0)
    assert new.points == base.points and new.last_seen == base.last_seen
    assert info.realised_fraction == 0.0 and info.n_affected == 0


def test_change_keeps_state_consistent():
    base = synthetic_state(50, 3, seed=5)
    new, info = apply_change(base, ChangeSpec(0.3, "distributed", "mixed"), 7, cell_size_m=1000.0)
    assert new.time_s == base.time_s and new.drones == base.drones and new.horizon_s == base.horizon_s
    assert set(new.last_seen) == set(new.points) == set(new.coverage)
    for i, p in new.points.items():
        if i in base.points and p == base.points[i]:
            assert new.last_seen[i] == base.last_seen[i]
        if i not in base.points:
            assert new.last_seen[i] == base.time_s
        assert p.revisit_s == 900 / p.risk
        assert all(j in new.points for j in new.coverage[i])
    assert info.n_after == len(new.points)


def test_concentrated_changes_are_more_clustered_than_distributed():
    base = synthetic_state(80, 4, seed=1)
    conc = [apply_change(base, ChangeSpec(0.15, "concentrated"), s, cell_size_m=1000.0)[1].dispersion for s in range(8)]
    dist = [apply_change(base, ChangeSpec(0.15, "distributed"), s, cell_size_m=1000.0)[1].dispersion for s in range(8)]
    assert max(conc) < 0.5 and min(dist) > 0.7
    assert sum(conc) / 8 < sum(dist) / 8


def test_change_is_deterministic_given_seed():
    base = synthetic_state(40, 3, seed=1)
    a = apply_change(base, ChangeSpec(0.2, "distributed"), 5, cell_size_m=1000.0)[0]
    b = apply_change(base, ChangeSpec(0.2, "distributed"), 5, cell_size_m=1000.0)[0]
    assert a.points == b.points


def test_bad_specs_rejected():
    for bad in (dict(fraction=1.0), dict(fraction=-0.1), dict(fraction=0.1, locality="x"), dict(fraction=0.1, kind="x")):
        try:
            ChangeSpec(**bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def _compare_cols(rows):
    return [(r["planner"], r["margin"], r["total_overdue"], r["overdue_count"], r["total_distance_m"]) for r in rows]


def test_run_pair_has_every_planner_equal_budgets_and_is_reproducible():
    base = synthetic_state(30, 3, seed=0)
    rows, traces = run_pair(base, ChangeSpec(0.3, "concentrated"), SMALL, seed=0)
    assert [r["planner"] for r in rows] == list(PLANNERS)
    budget = SMALL.ants * SMALL.iterations
    for r in rows:
        if r["planner"].startswith("aco_"):
            assert r["evaluations"] == budget
        if r["planner"] == "vns":
            assert r["evaluations"] <= budget
        assert math.isfinite(r["margin"]) and r["margin"] >= 1.0 or r["margin"] >= 0.0
        assert len({r["realised_fraction"]}) == 1
    assert len({r["realised_fraction"] for r in rows}) == 1
    rows2, _ = run_pair(base, ChangeSpec(0.3, "concentrated"), SMALL, seed=0)
    assert _compare_cols(rows) == _compare_cols(rows2)
    assert {t["planner"] for t in traces} == set(PLANNERS)


def test_no_change_means_all_four_strategies_agree():
    base = synthetic_state(30, 3, seed=1)
    rows, _ = run_pair(base, ChangeSpec(0.0), SMALL, seed=1)
    aco = [r for r in rows if r["planner"].startswith("aco_")]
    assert len(aco) == len(StrategyName)
    assert len({(r["margin"], r["total_overdue"], r["total_distance_m"]) for r in aco}) == 1


def test_strategies_differ_after_a_change():
    base = synthetic_state(40, 4, seed=2)
    seen = set()
    for seed in range(4):
        rows, _ = run_pair(base, ChangeSpec(0.4, "distributed"), SMALL, seed=seed)
        seen |= {(r["planner"], r["total_overdue"]) for r in rows if r["planner"].startswith("aco_")}
    assert len({p for p, _ in seen}) == 4 and len({v for _, v in seen}) > 4


def test_stats_pilot_filter_pairing_and_trace(tmp_path=None):
    import csv
    import os
    import tempfile

    d = tmp_path or tempfile.mkdtemp()
    path = os.path.join(d, "r.csv")
    fields = ["phase", "scenario", "trial", "target_fraction", "locality", "planner", "margin"]
    rows = []
    for trial in range(4):
        rows.append(dict(phase="final", scenario="s", trial=trial, target_fraction=0.1, locality="concentrated",
                         planner="aco_full_reset", margin=2.0))
        rows.append(dict(phase="final", scenario="s", trial=trial, target_fraction=0.1, locality="concentrated",
                         planner="aco_selective_reset", margin=1.5 if trial else 2.0))
        rows.append(dict(phase="pilot", scenario="s", trial=trial, target_fraction=0.1, locality="concentrated",
                         planner="aco_full_reset", margin=99.0))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    assert len(load_rows(path)) == 8 and len(load_rows(path, include_pilot=True)) == 12
    out = paired_summary(load_rows(path), "margin", "aco_full_reset")
    sel = next(r for r in out if r["planner"] == "aco_selective_reset")
    assert sel["n_pairs"] == 4 and math.isclose(sel["mean_diff"], -0.375)
    assert (sel["better"], sel["tied"], sel["worse"]) == (3, 1, 0)
    ts = trace_summary([dict(phase="final", scenario="s", trial=0, target_fraction=0.1, locality="c",
                             planner="vns", evaluations=e, best_margin=m)
                        for e, m in ((1, 3.0), (20, 2.5), (90, 2.0))], checkpoints=(10, 50, 100))
    assert [r["median_best_margin"] for r in ts] == [3.0, 2.5, 2.0]
