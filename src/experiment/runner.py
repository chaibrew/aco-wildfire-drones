"""Controlled-change experiment: one paired trial = one base scenario + one perturbation.

Protocol (matches the proposal's "Experimental design" table)
  1. Build the base PlanningState S0 (synthetic or captured from a FEDS replay).
  2. Warm up ONE ACO router on S0 for `warmup` replans, so it holds pheromone for S0.
     Warm-up is identical for every strategy (no change has occurred, so no strategy acts).
     The pre-change router state is snapshotted: every strategy starts from the SAME
     pheromone, tau_0 and RNG state ("identical pre-change pheromone").
  3. Apply one controlled change (magnitude x locality) to get S1. Same S1 for everyone.
  4. Every planner replans S1 with the same budget:
        greedy, vns (cold start), aco_retention, aco_full_reset,
        aco_fast_evaporation, aco_selective_reset (restored from the snapshot).
     All ACO variants and VNS use the same RNG seed ("paired trials").
  5. Every returned plan is re-scored by `fire_sim.simulation.evaluate` on S1, which also
     raises if a plan breaks the horizon rule, so no planner can cheat the objective.
"""
from __future__ import annotations

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Callable, Iterable

from fire_sim.simulation import PlanningState, evaluate as fire_sim_evaluate

from aco_router import ACOParams, ACORouter, StrategyName, diff_instances, state_to_instance
from baselines import GreedyRouter, VNSParams, VNSRouter

from .perturb import ChangeSpec, apply_change
from .scenarios import synthetic_state

STRATEGY_ORDER = (StrategyName.RETENTION, StrategyName.FULL_RESET, StrategyName.FAST_EVAPORATION,
                  StrategyName.SELECTIVE_RESET)
PLANNERS = ("greedy", "vns") + tuple(f"aco_{s.value}" for s in STRATEGY_ORDER)

RESULT_FIELDS = [
    "phase", "scenario", "trial", "target_fraction", "locality", "kind", "realised_fraction", "dispersion",
    "n_before", "n_after", "n_affected", "added", "removed", "changed", "planner",
    "objective", "margin", "total_overdue", "overdue_count", "mean_gap_s", "max_gap_s",
    "total_distance_m", "longest_route_m", "runtime_s", "evaluations", "iterations",
]
TRACE_FIELDS = ["phase", "scenario", "trial", "target_fraction", "locality", "planner", "evaluations",
                "elapsed_s", "best_objective", "best_margin", "best_total_overdue"]


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str = "final"  # "pilot" rows are written but excluded by the analysis script
    scenario: str = "synthetic"  # label written to the CSV
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60)
    localities: tuple[str, ...] = ("concentrated", "distributed")
    kind: str = "mixed"
    trials: int = 10
    first_seed: int = 0
    warmup: int = 5  # replans on the base state before the change
    ants: int = 10
    iterations: int = 15
    vns_evaluations: int | None = None  # None: ants * iterations (equal budget)
    n_points: int = 60  # synthetic only
    drones: int = 6
    cell_size_m: float = 1000.0
    sensor_radius_m: float | None = None
    workers: int = 1

    def aco_params(self) -> ACOParams:
        return ACOParams(n_ants=self.ants, max_iterations=self.iterations)

    def vns_params(self) -> VNSParams:
        return VNSParams(max_evaluations=self.vns_evaluations or self.ants * self.iterations)


def score_routes(state: PlanningState, routes: dict) -> dict:
    """Ground-truth scoring with the simulator's own evaluator (raises on infeasible routes)."""
    ev = fire_sim_evaluate(state, routes)
    ratios = {i: g / state.points[i].revisit_s for i, g in ev.gaps_s.items()}
    return {
        "objective": ev.objective,
        "margin": max(ratios.values(), default=0.0),
        "total_overdue": sum(max(0.0, r - 1.0) for r in ratios.values()),
        "overdue_count": ev.overdue_count,
        "mean_gap_s": ev.mean_gap_s,
        "max_gap_s": ev.max_gap_s,
        "total_distance_m": ev.total_distance_m,
        "longest_route_m": ev.longest_route_m,
    }


def warm_router(state: PlanningState, cfg: ExperimentConfig, seed: int) -> ACORouter:
    router = ACORouter(cfg.aco_params(), strategy=StrategyName.RETENTION, seed=seed)
    instance = state_to_instance(state)
    for _ in range(max(1, cfg.warmup)):
        router.plan(instance)
    return router


def run_pair(base: PlanningState, spec: ChangeSpec, cfg: ExperimentConfig, seed: int,
             label: str = "") -> tuple[list[dict], list[dict]]:
    """Run every planner on one (base, change). Returns (result rows, trace rows)."""
    warmed = warm_router(base, cfg, seed)
    snap = warmed.snapshot()
    changed, info = apply_change(base, spec, seed + 10_000, cell_size_m=cfg.cell_size_m,
                                 sensor_radius_m=cfg.sensor_radius_m)
    inst0, inst1 = state_to_instance(base), state_to_instance(changed)
    change = diff_instances(inst0, inst1)
    replan_seed = seed + 20_000

    context = {"phase": cfg.phase, "scenario": label or cfg.scenario, "trial": seed,
               "target_fraction": spec.fraction, "locality": spec.locality, "kind": spec.kind,
               "realised_fraction": info.realised_fraction, "dispersion": info.dispersion,
               "n_before": info.n_before, "n_after": info.n_after, "n_affected": info.n_affected,
               "added": info.added, "removed": info.removed, "changed": info.changed}
    rows: list[dict] = []
    traces: list[dict] = []

    def record(name: str, routes: dict, runtime: float, evaluations: int, iterations: int, trace) -> None:
        rows.append({**context, "planner": name, **score_routes(changed, routes), "runtime_s": runtime,
                     "evaluations": evaluations, "iterations": iterations})
        for t in trace:
            traces.append({**{k: context[k] for k in ("phase", "scenario", "trial", "target_fraction", "locality")},
                           "planner": name, "evaluations": t[0], "elapsed_s": t[1], "best_objective": t[2],
                           "best_margin": t[3], "best_total_overdue": t[4]})

    t0 = time.perf_counter()
    routes = GreedyRouter()(changed)
    record("greedy", routes, time.perf_counter() - t0, 1, 0, [(1, time.perf_counter() - t0, *_obj(changed, routes))])

    vns = VNSRouter(cfg.vns_params(), seed=replan_seed)
    res = vns.plan(changed)
    record("vns", res.routes, res.runtime_s, res.evaluations, res.iterations,
           [(t.iteration, t.elapsed_s, t.best_objective, t.best_margin, t.best_total_overdue) for t in res.trace])

    for strategy in STRATEGY_ORDER:
        router = ACORouter(cfg.aco_params(), strategy=strategy, seed=replan_seed)
        router.restore(snap)
        router.reseed(replan_seed)  # identical random stream for every strategy
        result = router.plan(inst1, change)
        record(f"aco_{strategy.value}", result.plan.for_drones(inst1.drone_ids), result.runtime_s,
               result.iterations * cfg.ants, result.iterations,
               [(t.iteration * cfg.ants, t.elapsed_s, t.best_objective, t.best_margin, t.best_total_overdue)
                for t in result.trace])
    return rows, traces


def _obj(state: PlanningState, routes: dict) -> tuple[float, float, float]:
    s = score_routes(state, routes)
    return s["objective"], s["margin"], s["total_overdue"]


def _trial_job(args):
    cfg, seed, fraction, locality = args
    base = synthetic_state(cfg.n_points, cfg.drones, seed, cell_size_m=cfg.cell_size_m,
                           sensor_radius_m=cfg.sensor_radius_m)
    return run_pair(base, ChangeSpec(fraction, locality, cfg.kind), cfg, seed)


def run_synthetic(cfg: ExperimentConfig, progress: Callable[[str], None] = print):
    jobs = [(cfg, s, f, loc) for s in range(cfg.first_seed, cfg.first_seed + cfg.trials)
            for f, loc in product(cfg.fractions, cfg.localities)]
    rows: list[dict] = []
    traces: list[dict] = []
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as pool:
            outputs = pool.map(_trial_job, jobs)
            for job, (r, t) in zip(jobs, outputs):
                rows += r
                traces += t
                progress(f"seed {job[1]} fraction {job[2]:.2f} {job[3]} done")
    else:
        for job in jobs:
            r, t = _trial_job(job)
            rows += r
            traces += t
            progress(f"seed {job[1]} fraction {job[2]:.2f} {job[3]} done")
    return rows, traces


def run_from_states(bases: dict[str, PlanningState], cfg: ExperimentConfig, progress: Callable[[str], None] = print):
    """Same protocol on real captured states, e.g. {'camp3885_stage40': state, ...}."""
    rows: list[dict] = []
    traces: list[dict] = []
    for label, base in bases.items():
        for seed in range(cfg.first_seed, cfg.first_seed + cfg.trials):
            for f, loc in product(cfg.fractions, cfg.localities):
                r, t = run_pair(base, ChangeSpec(f, loc, cfg.kind), cfg, seed, label=label)
                rows += r
                traces += t
            progress(f"{label} seed {seed} done")
    return rows, traces


def write_csv(path: str, fields: list[str], rows: Iterable[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
