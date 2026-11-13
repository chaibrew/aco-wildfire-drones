"""Running the tuning steps: parallel evaluation, budget check, races, confirmation, CSV output."""
from __future__ import annotations

import csv
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from aco_router import ACOParams, StrategyName

from .frace import RaceResult
from .instances import STRATEGIES, TuningInstance, rank_key, strategy_costs
from .space import params_to_dict

MEASURES = ("margin", "total_overdue", "overdue_count")
TIE = 1e-9


def _job(args):
    params, inst, strategies, warmup = args
    return strategy_costs(params, inst, strategies, warmup=warmup)


class ParallelEvaluator:
    """evaluate(list of ACOParams, instance index) -> list of strategy_costs results, run in parallel."""

    def __init__(self, instances: Sequence[TuningInstance], workers: int = 1,
                 strategies: Sequence[StrategyName] = STRATEGIES, warmup: int = 5):
        self.instances = list(instances)
        self.strategies = tuple(strategies)
        self.warmup = warmup
        self.pool = ProcessPoolExecutor(workers) if workers > 1 else None

    def __call__(self, params: Sequence[ACOParams], inst: int) -> list[dict]:
        jobs = [(p, self.instances[inst], self.strategies, self.warmup) for p in params]
        if self.pool is None:
            return [_job(j) for j in jobs]
        return list(self.pool.map(_job, jobs))

    def across_instances(self, params: ACOParams, instances: Sequence[int] | None = None) -> list[dict]:
        """One setting on many instances (in parallel); results in instance order."""
        idx = range(len(self.instances)) if instances is None else instances
        jobs = [(params, self.instances[i], self.strategies, self.warmup) for i in idx]
        if self.pool is None:
            return [_job(j) for j in jobs]
        return list(self.pool.map(_job, jobs))

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------- rows and CSV

def result_rows(instances: Sequence[TuningInstance], inst: int, setting: str | int, result: dict,
                **extra) -> list[dict]:
    ti = instances[inst]
    return [{"instance": inst + 1, "seed": ti.seed, "base": ti.label, "n_points": len(ti.base.points),
             "fraction": ti.change.fraction, "locality": ti.change.locality, "setting": setting, **extra,
             "strategy": strategy, **{m: s[m] for m in MEASURES}, "seconds": s["seconds"]}
            for strategy, s in result.items()]


def write_csv(path: str, rows: Sequence[dict], fields: Sequence[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = list(fields or (rows[0].keys() if rows else []))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- paired comparisons

def compare(a: Sequence[float], b: Sequence[float]) -> dict:
    """Paired comparison of b against a (lower is better): counts of b better / tied / worse, mean change,
    and a two-sided Wilcoxon signed-rank p-value (zsplit for ties; 1.0 when every pair is tied)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    diff = b - a
    better, worse = int((diff < -TIE).sum()), int((diff > TIE).sum())
    tied = len(diff) - better - worse
    p = 1.0 if better + worse == 0 else float(stats.wilcoxon(a, b, zero_method="zsplit").pvalue)
    return {"n": len(diff), "better": better, "tied": tied, "worse": worse, "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "mean_change": float(diff.mean()), "p": p}


def paired(rows: Sequence[dict], ref: str, other: str, key: Callable[[dict], float],
           strategy: str | None = None) -> dict:
    """Pairs rows of setting `ref` and `other` by (instance, strategy)."""
    def index(name):
        return {(r["instance"], r["strategy"]): key(r) for r in rows
                if str(r["setting"]) == name and (strategy is None or r["strategy"] == strategy)}
    ia, ib = index(ref), index(other)
    common = sorted(set(ia) & set(ib))
    return compare([ia[k] for k in common], [ib[k] for k in common])


def _f(r: dict, m: str) -> float:
    return float(r[m])


def rank_value(r: dict) -> float:
    return rank_key({m: float(r[m]) for m in MEASURES})


def budget_summary(rows: Sequence[dict], budgets: Sequence[tuple[int, int]]) -> list[dict]:
    out = []
    for i, (ants, iters) in enumerate(budgets):
        name = f"{ants}x{iters}"
        mine = [r for r in rows if r["setting"] == name]
        row = {"budget": name, "ants": ants, "iterations": iters, "evaluations": ants * iters, "cases": len(mine),
               **{m: float(np.mean([_f(r, m) for r in mine])) for m in MEASURES},
               "seconds_per_plan": float(np.mean([_f(r, "seconds") for r in mine]))}
        if i + 1 < len(budgets):
            nxt = f"{budgets[i + 1][0]}x{budgets[i + 1][1]}"
            c = paired(rows, name, nxt, rank_value)
            t = paired(rows, name, nxt, lambda r: _f(r, "total_overdue"))
            row.update({"next": nxt, "next_better": c["better"], "next_tied": c["tied"], "next_worse": c["worse"],
                        "p_total_overdue_vs_next": t["p"]})
        out.append(row)
    return out


def confirmation_summary(rows: Sequence[dict], ref: str, others: Sequence[str]) -> list[dict]:
    """One row per (other setting, strategy or 'all', measure) comparing it with `ref`."""
    strategies = [None] + sorted({r["strategy"] for r in rows})
    out = []
    for other in others:
        for strategy in strategies:
            for measure in ("rank_key",) + MEASURES:
                key = rank_value if measure == "rank_key" else (lambda r, m=measure: _f(r, m))
                c = paired(rows, ref, other, key, strategy)
                out.append({"reference": ref, "setting": other, "strategy": strategy or "all",
                            "measure": measure, **c})
    return out


def setting_means(rows: Sequence[dict]) -> list[dict]:
    names = list(dict.fromkeys(str(r["setting"]) for r in rows))
    out = []
    for name in names:
        mine = [r for r in rows if str(r["setting"]) == name]
        out.append({"setting": name, "cases": len(mine),
                    **{m: float(np.mean([_f(r, m) for r in mine])) for m in MEASURES}})
    return out


# ---------------------------------------------------------------- race output

def race_candidate_rows(race_result: RaceResult, candidates: Sequence[ACOParams], names: Sequence[str],
                        labels: dict[int, str] | None = None) -> list[dict]:
    final_ranks = race_result.mean_rank()
    rows = []
    for c, params in enumerate(candidates):
        d = params_to_dict(params)
        rows.append({"candidate": c, "label": (labels or {}).get(c, ""),
                     **{n: d[n] for n in names},
                     "eliminated_after": race_result.eliminated_at.get(c, ""),
                     "instances_raced": race_result.eliminated_at.get(c, race_result.instances_used),
                     "survived": int(c in race_result.alive),
                     "final_mean_rank": final_ranks.get(c, ""),
                     "chosen": int(c == race_result.best)})
    return rows


def race_log_rows(race_result: RaceResult) -> list[dict]:
    return [{"instance": s.instance, "alive_before": s.alive_before, "alive_after": s.alive_after,
             "test": s.test, "p_value": s.p_value, "eliminated": " ".join(map(str, s.eliminated))}
            for s in race_result.steps]
