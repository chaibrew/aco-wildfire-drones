from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from aco_router import ACOParams, ACORouter, StrategyName, diff_instances, state_to_instance
from experiment.perturb import ChangeSpec, apply_change
from experiment.runner import score_routes
from experiment.scenarios import captured_states, synthetic_state
from fire_sim.simulation import PlanningState

FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60)
LOCALITIES = ("concentrated", "distributed")
SYNTHETIC_POINTS = 137
STRATEGIES = tuple(StrategyName)


@dataclass(frozen=True)
class SeedPlan:
    first_seed: int
    fire_seed: int
    stages: tuple[int, ...]


# Must stay disjoint from each other and from the reported experiments (seeds from 0).
SEEDS: dict[str, SeedPlan] = {
    "shared": SeedPlan(1000, 1000, (20, 60, 100)),
    "irace": SeedPlan(1200, 1200, (25, 65, 105)),
    "fast-evaporation": SeedPlan(1500, 1500, (20, 60, 100)),
    "budget": SeedPlan(2000, 2000, (20, 60, 100)),
    "confirm": SeedPlan(2100, 2100, (30, 70, 110)),
}
REPORTED_SEEDS = range(0, 1000)


@dataclass(frozen=True)
class TuningInstance:
    seed: int
    label: str
    base: PlanningState
    change: ChangeSpec


def camp_states(seeds: SeedPlan, *, gpkg: str = "data/raw/LargeFires_2018.gpkg", fire_id: int = 3885,
                year: int = 2018, cache: str = "data/cache", drones: int = 6) -> dict[str, PlanningState]:
    states = captured_states(gpkg, fire_id, year, seed=seeds.fire_seed, cache=cache, drones=drones,
                             stages=seeds.stages)
    return {f"camp_stage{k}": s for k, s in sorted(states.items())}


def make_instances(seeds: SeedPlan, n: int, camp: Mapping[str, PlanningState] | None = None, *,
                   drones: int = 6, n_points: int = SYNTHETIC_POINTS) -> list[TuningInstance]:
    if seeds.first_seed in REPORTED_SEEDS:
        raise ValueError("tuning seeds must not overlap the reported experiments' seeds")
    camp_items = list((camp or {}).items())
    out = []
    for i in range(n):
        seed = seeds.first_seed + i
        rng = random.Random(seed)
        change = ChangeSpec(rng.choice(FRACTIONS), rng.choice(LOCALITIES))
        if camp_items and i % 2 == 1:
            label, base = camp_items[(i // 2) % len(camp_items)]
        else:
            label, base = f"synthetic{n_points}", synthetic_state(n_points, drones, seed)
        out.append(TuningInstance(seed, label, base, change))
    return out


def plan_scores(state: PlanningState, routes: dict) -> dict[str, float]:
    s = score_routes(state, routes)
    return {"margin": s["margin"], "total_overdue": s["total_overdue"], "overdue_count": s["overdue_count"]}


def rank_key(scores: Mapping[str, float]) -> float:
    return round(scores["margin"], 6) * 1e6 + scores["total_overdue"]


def strategy_costs(params: ACOParams, inst: TuningInstance, strategies: Sequence[StrategyName] = STRATEGIES, *,
                   warmup: int = 5, cell_size_m: float = 1000.0) -> dict[str, dict[str, float]]:
    warm = ACORouter(params, strategy=StrategyName.RETENTION, seed=inst.seed)
    inst0 = state_to_instance(inst.base)
    for _ in range(max(1, warmup)):
        warm.plan(inst0)
    snap = warm.snapshot()
    changed, _ = apply_change(inst.base, inst.change, inst.seed + 10_000, cell_size_m=cell_size_m)
    inst1 = state_to_instance(changed)
    change = diff_instances(inst0, inst1)
    out = {}
    for strategy in strategies:
        router = ACORouter(params, strategy=strategy, seed=inst.seed)
        router.restore(snap)
        router.reseed(inst.seed + 20_000)  # same random stream for every strategy (paired comparison)
        result = router.plan(inst1, change)
        scores = plan_scores(changed, result.plan.for_drones(inst1.drone_ids))
        out[StrategyName(strategy).value] = {**scores, "seconds": result.runtime_s}
    return out


def block_scores(results: Sequence[Mapping[str, Mapping[str, float]]]) -> list[float]:
    strategies = list(results[0])
    keys = np.array([[rank_key(r[s]) for s in strategies] for r in results], dtype=float)
    return list(np.apply_along_axis(rankdata, 0, keys).mean(axis=1))
