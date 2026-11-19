from __future__ import annotations

import csv
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.stats import rankdata

from aco_router import ACOParams, ACORouter, BudgetKind, StrategyName, Tau0Mode, diff_instances, state_to_instance
from experiment.perturb import ChangeSpec, apply_change
from experiment.runner import score_routes
from experiment.scenarios import captured_states, synthetic_state
from fire_sim.simulation import PlanningState

from .racing import Parameter, ParameterSpace, RaceResult

SHARED_SPACE = ParameterSpace((
    Parameter("alpha", 0.5, 2.0),
    Parameter("beta", 0.5, 3.0),
    Parameter("urgency_weight", 0.5, 4.0),
    Parameter("rho", 0.02, 0.5, log=True),
    Parameter("tau_min_ratio", 0.001, 0.2, log=True),
    Parameter("urgency_floor", 0.01, 0.3, log=True),
))
FAST_EVAPORATION_NAMES = ("rho_high", "fast_evaporation_iterations")
RHO_HIGH_GRID = (0.2, 0.3, 0.5, 0.7, 0.9)
K_GRID = (2, 5, 10, 20)

# Fixed copy of the pre-tuning values: "defaults" must still mean untuned after config.py changed.
UNTUNED_PARAMS = ACOParams(n_ants=10, max_iterations=15, alpha=1.0, beta=1.0, rho=0.1, urgency_weight=2.0,
                           urgency_floor=0.05, tau_min_ratio=0.01, rho_high=0.5, fast_evaporation_iterations=10)

FRACTIONS = (0.05, 0.10, 0.20, 0.40, 0.60)
LOCALITIES = ("concentrated", "distributed")
SYNTHETIC_POINTS = 137
STRATEGIES = tuple(StrategyName)
MEASURES = ("margin", "total_overdue", "overdue_count")
CHANGE_SEED_OFFSET = 10_000
REPLAN_SEED_OFFSET = 20_000
TIE = 1e-9


def sobol_candidates(base: ACOParams, n: int = 32, seed: int = 0) -> list[ACOParams]:
    return [base] + [base.with_(**v) for v in SHARED_SPACE.sobol(n, seed)]


def fast_evaporation_grid(base: ACOParams, rho_highs=RHO_HIGH_GRID, ks=K_GRID) -> list[ACOParams]:
    current = (base.rho_high, base.fast_evaporation_iterations)
    return [base] + [base.with_(rho_high=rh, fast_evaporation_iterations=k)
                     for rh in rho_highs if rh > base.rho for k in ks if (rh, k) != current]


def params_to_dict(params: ACOParams) -> dict:
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(params).items()}


def params_from_dict(d: dict, base: ACOParams | None = None) -> ACOParams:
    unknown = set(d) - {f.name for f in fields(ACOParams)}
    if unknown:
        raise ValueError(f"unknown ACOParams fields: {sorted(unknown)}")
    vals = dict(d)
    if "budget_kind" in vals:
        vals["budget_kind"] = BudgetKind(vals["budget_kind"])
    if "tau0_mode" in vals:
        vals["tau0_mode"] = Tau0Mode(vals["tau0_mode"])
    return (base or ACOParams()).with_(**vals)


def save_params(path: str, params: ACOParams, **meta) -> None:
    with open(path, "w") as fh:
        json.dump({"params": params_to_dict(params), **meta}, fh, indent=2)
        fh.write("\n")


def load_params(path: str, base: ACOParams | None = None) -> ACOParams:
    with open(path) as fh:
        d = json.load(fh)
    return params_from_dict(d.get("params", d), base)


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

    def columns(self) -> dict:
        return {"seed": self.seed, "base": self.label, "n_points": len(self.base.points),
                "fraction": self.change.fraction, "locality": self.change.locality}


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


def rank_key(margin: float, total_overdue: float) -> float:
    return round(margin, 6) * 1e6 + total_overdue


@dataclass(frozen=True)
class Scores:
    margin: float
    total_overdue: float
    overdue_count: float
    seconds: float = 0.0

    @property
    def rank_key(self) -> float:
        return rank_key(self.margin, self.total_overdue)

    @classmethod
    def of_plan(cls, state: PlanningState, routes: dict, seconds: float = 0.0) -> "Scores":
        s = score_routes(state, routes)
        return cls(s["margin"], s["total_overdue"], s["overdue_count"], seconds)

    def measures(self) -> dict[str, float]:
        return {m: getattr(self, m) for m in MEASURES}


StrategyScores = dict[str, Scores]


def score_setting(params: ACOParams, inst: TuningInstance, strategies: Sequence[StrategyName] = STRATEGIES, *,
                  warmup: int = 5, cell_size_m: float = 1000.0) -> StrategyScores:
    warm = ACORouter(params, strategy=StrategyName.RETENTION, seed=inst.seed)
    inst0 = state_to_instance(inst.base)
    for _ in range(max(1, warmup)):
        warm.plan(inst0)
    snap = warm.snapshot()
    changed, _ = apply_change(inst.base, inst.change, inst.seed + CHANGE_SEED_OFFSET, cell_size_m=cell_size_m)
    inst1 = state_to_instance(changed)
    change = diff_instances(inst0, inst1)
    out = {}
    for strategy in strategies:
        router = ACORouter(params, strategy=strategy, seed=inst.seed)
        router.restore(snap)
        router.reseed(inst.seed + REPLAN_SEED_OFFSET)  # same random stream for every strategy (paired comparison)
        result = router.plan(inst1, change)
        out[StrategyName(strategy).value] = Scores.of_plan(changed, result.plan.for_drones(inst1.drone_ids),
                                                           result.runtime_s)
    return out


def mean_strategy_rank(results: Sequence[StrategyScores]) -> list[float]:
    strategies = list(results[0])
    keys = np.array([[r[s].rank_key for s in strategies] for r in results], dtype=float)
    return list(np.apply_along_axis(rankdata, 0, keys).mean(axis=1))


def _score_job(args) -> StrategyScores:
    params, inst, strategies, warmup = args
    return score_setting(params, inst, strategies, warmup=warmup)


class ParallelEvaluator:
    def __init__(self, instances: Sequence[TuningInstance], workers: int = 1,
                 strategies: Sequence[StrategyName] = STRATEGIES, warmup: int = 5):
        self.instances = list(instances)
        self.strategies = tuple(strategies)
        self.warmup = warmup
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None

    def on_instance(self, settings: Sequence[ACOParams], inst: int) -> list[StrategyScores]:
        return self._map([(p, self.instances[inst]) for p in settings])

    def across_instances(self, params: ACOParams) -> list[StrategyScores]:
        return self._map([(params, inst) for inst in self.instances])

    def _map(self, pairs) -> list[StrategyScores]:
        jobs = [(p, inst, self.strategies, self.warmup) for p, inst in pairs]
        return list(self.pool.map(_score_job, jobs)) if self.pool else [_score_job(j) for j in jobs]

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def __enter__(self) -> "ParallelEvaluator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def result_rows(instances: Sequence[TuningInstance], inst: int, setting: str | int, scores: StrategyScores,
                **extra) -> list[dict]:
    head = {"instance": inst + 1, **instances[inst].columns(), "setting": setting, **extra}
    return [{**head, "strategy": strategy, **s.measures(), "seconds": s.seconds} for strategy, s in scores.items()]


def write_csv(path: str, rows: Sequence[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)


def compare(a: Sequence[float], b: Sequence[float]) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    better, worse = int((diff < -TIE).sum()), int((diff > TIE).sum())
    p = 1.0 if better + worse == 0 else float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    return {"n": len(diff), "better": better, "tied": len(diff) - better - worse, "worse": worse,
            "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_change": float(diff.mean()), "p": p}


def measure_key(measure: str) -> Callable[[dict], float]:
    if measure == "rank_key":
        return lambda r: rank_key(float(r["margin"]), float(r["total_overdue"]))
    return lambda r: float(r[measure])


def paired(rows: Sequence[dict], ref: str, other: str, measure: str, strategy: str | None = None) -> dict:
    key = measure_key(measure)

    def index(name: str) -> dict:
        return {(r["instance"], r["strategy"]): key(r) for r in rows
                if str(r["setting"]) == name and strategy in (None, r["strategy"])}

    ia, ib = index(ref), index(other)
    common = sorted(set(ia) & set(ib))
    return compare([ia[k] for k in common], [ib[k] for k in common])


def setting_means(rows: Sequence[dict]) -> list[dict]:
    out = []
    for name in dict.fromkeys(str(r["setting"]) for r in rows):
        mine = [r for r in rows if str(r["setting"]) == name]
        out.append({"setting": name, "cases": len(mine),
                    **{m: float(np.mean([float(r[m]) for r in mine])) for m in MEASURES}})
    return out


def budget_summary(rows: Sequence[dict], budgets: Sequence[tuple[int, int]]) -> list[dict]:
    names = [f"{a}x{i}" for a, i in budgets]
    out = []
    for i, ((ants, iters), name) in enumerate(zip(budgets, names)):
        mine = [r for r in rows if r["setting"] == name]
        row = {"budget": name, "ants": ants, "iterations": iters, "evaluations": ants * iters, "cases": len(mine),
               **{m: float(np.mean([float(r[m]) for r in mine])) for m in MEASURES},
               "seconds_per_plan": float(np.mean([float(r["seconds"]) for r in mine]))}
        if i + 1 < len(names):
            c = paired(rows, name, names[i + 1], "rank_key")
            row.update({"next": names[i + 1], "next_better": c["better"], "next_tied": c["tied"],
                        "next_worse": c["worse"],
                        "p_total_overdue_vs_next": paired(rows, name, names[i + 1], "total_overdue")["p"]})
        out.append(row)
    return out


def confirmation_summary(rows: Sequence[dict], ref: str, others: Sequence[str]) -> list[dict]:
    strategies = [None] + sorted({r["strategy"] for r in rows})
    return [{"reference": ref, "setting": other, "strategy": strategy or "all", "measure": measure,
             **paired(rows, ref, other, measure, strategy)}
            for other in others for strategy in strategies for measure in ("rank_key",) + MEASURES]


def race_candidate_rows(result: RaceResult, candidates: Sequence[ACOParams], names: Sequence[str],
                        labels: Mapping[int, str]) -> list[dict]:
    final_ranks = result.mean_rank()
    return [{"candidate": c, "label": labels.get(c, ""), **{n: params_to_dict(p)[n] for n in names},
             "eliminated_after": result.eliminated_at.get(c, ""),
             "instances_raced": result.eliminated_at.get(c, result.instances_used),
             "survived": int(c in result.alive), "final_mean_rank": final_ranks.get(c, ""),
             "chosen": int(c == result.best)}
            for c, p in enumerate(candidates)]


def race_log_rows(result: RaceResult) -> list[dict]:
    return [{"instance": s.instance, "alive_before": s.alive_before, "alive_after": s.alive_after,
             "test": s.test, "p_value": s.p_value, "eliminated": " ".join(map(str, s.eliminated))}
            for s in result.steps]
