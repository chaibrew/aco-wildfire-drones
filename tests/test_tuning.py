import json

import numpy as np
import pytest

from aco_router import ACOParams, StrategyName
from experiment.perturb import ChangeSpec
from tuning import (SEEDS, SHARED_SPACE, SeedPlan, TuningInstance, block_scores, fast_evaporation_grid,
                    friedman_survivors, iterated_race, load_params, make_instances, race, rank_key, save_params,
                    sobol_candidates, strategy_costs, to_unit, from_unit)
from tuning.run import compare, paired

TINY = ACOParams(n_ants=3, max_iterations=2)


def _race_on(costs, **kw):
    costs = np.asarray(costs, float)
    return race(costs.shape[1], costs.shape[0], lambda alive, i: [costs[i, c] for c in alive], **kw)


def test_race_drops_clearly_worse_and_keeps_best():
    rng = np.random.default_rng(0)
    costs = rng.normal(0, 0.1, (20, 5)) + np.array([0.0, 0.05, 3.0, 4.0, 5.0])
    r = _race_on(costs)
    assert 0 in r.alive
    assert {2, 3, 4}.isdisjoint(r.alive)
    assert r.best in (0, 1)
    assert all(v >= 5 for v in r.eliminated_at.values())


def test_race_keeps_everyone_when_equal():
    r = _race_on(np.ones((10, 4)))
    assert r.alive == [0, 1, 2, 3]
    assert r.instances_used == 10


def test_no_test_before_first_test():
    costs = np.tile([0.0, 10.0, 20.0], (10, 1))
    seen = []
    r = _race_on(costs, first_test=7, progress=seen.append)
    assert all(s.test == "none" for s in r.steps[:6])
    assert min(r.eliminated_at.values()) >= 7


def test_two_candidates_use_wilcoxon():
    costs = np.column_stack([np.zeros(12), np.arange(1, 13)])
    keep, p, test = friedman_survivors(costs, 0.05)
    assert test == "wilcoxon" and keep == [0] and p < 0.05


def test_race_stop_hook_ends_early():
    r = _race_on(np.ones((10, 3)), stop=lambda alive, i: i >= 6)
    assert r.instances_used == 6


def test_sobol_starts_with_defaults_stays_in_range_and_is_reproducible():
    base = ACOParams()
    a, b = sobol_candidates(base, 16, 0), sobol_candidates(base, 16, 0)
    assert a[0] == base and a == b and len(a) == 17
    for c in a[1:]:
        for name, lo, hi, _ in SHARED_SPACE:
            assert lo <= getattr(c, name) <= hi
    assert sobol_candidates(base, 16, 1)[1:] != a[1:]


def test_unit_round_trip():
    u = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 0.3])
    assert np.allclose(to_unit(from_unit(u)), u, atol=1e-3)


def test_fast_evaporation_grid_starts_with_current_and_skips_low_rho_high():
    base = ACOParams(rho=0.36, rho_high=0.5, fast_evaporation_iterations=10)
    grid = fast_evaporation_grid(base)
    assert grid[0] == base
    assert all(c.rho_high > base.rho for c in grid)
    pairs = [(c.rho_high, c.fast_evaporation_iterations) for c in grid]
    assert len(pairs) == len(set(pairs)) == 3 * 4


def test_instances_use_tuning_seeds_and_are_reproducible():
    plan = SeedPlan(1000, 1000, (20,))
    a, b = make_instances(plan, 4), make_instances(plan, 4)
    assert [i.seed for i in a] == [1000, 1001, 1002, 1003]
    assert [(i.change, len(i.base.points)) for i in a] == [(i.change, len(i.base.points)) for i in b]
    assert all(len(i.base.points) == 137 for i in a)
    with pytest.raises(ValueError):
        make_instances(SeedPlan(0, 0, (6,)), 2)


def test_seed_blocks_are_disjoint():
    blocks = [range(s.first_seed, s.first_seed + 100) for s in SEEDS.values()]
    for i, a in enumerate(blocks):
        assert a.start >= 1000
        for b in blocks[i + 1:]:
            assert not set(a) & set(b)


def test_instances_alternate_with_camp_states():
    synth = make_instances(SeedPlan(1000, 1000, (20,)), 1)[0].base
    inst = make_instances(SeedPlan(1000, 1000, (20,)), 4, {"camp_stage20": synth})
    assert [i.label for i in inst] == ["synthetic137", "camp_stage20", "synthetic137", "camp_stage20"]


@pytest.fixture(scope="module")
def instance():
    return make_instances(SeedPlan(1000, 1000, (20,)), 1)[0]


def test_strategy_costs_reproducible_and_covers_every_strategy(instance):
    a = strategy_costs(TINY, instance, warmup=1)
    b = strategy_costs(TINY, instance, warmup=1)
    assert set(a) == {s.value for s in StrategyName}
    strip = lambda d: {k: {m: v for m, v in s.items() if m != "seconds"} for k, s in d.items()}  # noqa: E731
    assert strip(a) == strip(b)


def test_zero_change_gives_identical_scores(instance):
    same = TuningInstance(instance.seed, instance.label, instance.base, ChangeSpec(0.0))
    out = strategy_costs(TINY, same, warmup=1)
    keys = {rank_key(s) for s in out.values()}
    assert len(keys) == 1


def test_rank_key_orders_by_margin_then_total_overdue():
    a = {"margin": 2.0, "total_overdue": 100.0}
    b = {"margin": 2.1, "total_overdue": 0.0}
    c = {"margin": 2.0 + 1e-9, "total_overdue": 50.0}
    assert rank_key(a) < rank_key(b)
    assert rank_key(c) < rank_key(a)


def test_block_scores_rank_within_block():
    def r(m, o):
        return {"margin": m, "total_overdue": o}
    results = [{"retention": r(1, 5), "full_reset": r(1, 1)},
               {"retention": r(1, 1), "full_reset": r(1, 3)},
               {"retention": r(2, 0), "full_reset": r(2, 0)}]
    assert block_scores(results) == [1.5, 1.5, 3.0]


def test_iterated_race_finds_good_region():
    space = SHARED_SPACE[:2]
    target = np.array([0.8, 0.2])

    def to_values(u):
        return {"x": float(u[0]), "y": float(u[1])}

    def to_u(v):
        return np.array([v["x"], v["y"]])

    def evaluate(configs, inst):
        noise = np.random.default_rng(inst).normal(0, 0.01)
        return [float(np.sum((to_u(c) - target) ** 2)) + noise for c in configs]

    res = iterated_race(space, to_values, to_u, evaluate, 20, budget=300, seed=0)
    assert res.experiments <= 300
    best = to_u(res.configs[res.best])
    assert np.sum((best - target) ** 2) < 0.05
    assert len(res.iterations) >= 2


def test_params_json_round_trip(tmp_path):
    p = ACOParams(alpha=0.73, beta=2.49, rho=0.36)
    path = tmp_path / "p.json"
    save_params(str(path), p, note="x")
    assert load_params(str(path)) == p
    assert json.loads(path.read_text())["note"] == "x"
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({"beta": 2.0}))
    assert load_params(str(flat)).beta == 2.0


def test_compare_counts_and_pairs():
    c = compare([1, 2, 3, 4], [0, 2, 4, 3])
    assert (c["better"], c["tied"], c["worse"]) == (2, 1, 1)
    rows = [{"instance": i, "strategy": "s", "setting": n, "v": v}
            for i, (a, b) in enumerate([(1, 0), (2, 1), (3, 2)]) for n, v in (("a", a), ("b", b))]
    assert paired(rows, "a", "b", lambda r: r["v"])["better"] == 3
