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
    "phase", "scenario", "trial", "target_fraction", "locality", "kind", "direction", "realised_fraction",
    "dispersion", "n_before", "n_after", "n_affected", "added", "removed", "changed", "reset_points", "planner",
    "objective", "margin", "total_overdue", "overdue_count", "mean_gap_s", "max_gap_s",
    "total_distance_m", "longest_route_m", "runtime_s", "evaluations", "iterations",
]
TRACE_FIELDS = ["phase", "scenario", "trial", "target_fraction", "locality", "planner", "evaluations",
                "elapsed_s", "best_objective", "best_margin", "best_total_overdue"]


@dataclass(frozen=True)
class ExperimentConfig:
    phase: str = "final"
    scenario: str = "synthetic"
    fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40, 0.60)
    localities: tuple[str, ...] = ("concentrated", "distributed")
    kind: str = "mixed"
    direction: str = "cycle"
    trials: int = 10
    first_seed: int = 0
    warmup: int = 5
    ants: int = 20
    iterations: int = 60
    vns_evaluations: int | None = None
    n_points: int = 60
    drones: int = 6
    cell_size_m: float = 1000.0
    sensor_radius_m: float | None = None
    workers: int = 1
    params: ACOParams | None = None
    selective_radius_m: float | None = None
    planners: tuple[str, ...] = PLANNERS

    def __post_init__(self) -> None:
        unknown = set(self.planners) - set(PLANNERS)
        if unknown:
            raise ValueError(f"unknown planners {sorted(unknown)}; choose from {PLANNERS}")

    def aco_params(self) -> ACOParams:
        p = (self.params or ACOParams()).with_(n_ants=self.ants, max_iterations=self.iterations)
        return p if self.selective_radius_m is None else p.with_(selective_radius_m=self.selective_radius_m)

    def vns_params(self) -> VNSParams:
        return VNSParams(max_evaluations=self.vns_evaluations or self.ants * self.iterations)


def score_routes(state: PlanningState, routes: dict) -> dict:
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
    run_aco = any(n.startswith("aco_") for n in cfg.planners)
    snap = warm_router(base, cfg, seed).snapshot() if run_aco else None
    changed, info = apply_change(base, spec, seed + 10_000, cell_size_m=cfg.cell_size_m,
                                 sensor_radius_m=cfg.sensor_radius_m)
    inst0, inst1 = state_to_instance(base), state_to_instance(changed)
    change = diff_instances(inst0, inst1)
    replan_seed = seed + 20_000

    context = {"phase": cfg.phase, "scenario": label or cfg.scenario, "trial": seed,
               "target_fraction": spec.fraction, "locality": spec.locality, "kind": spec.kind,
               "direction": spec.direction, "realised_fraction": info.realised_fraction, "dispersion": info.dispersion,
               "n_before": info.n_before, "n_after": info.n_after, "n_affected": info.n_affected,
               "added": info.added, "removed": info.removed, "changed": info.changed}
    rows: list[dict] = []
    traces: list[dict] = []

    def record(name: str, routes: dict, runtime: float, evaluations: int, iterations: int, trace,
               reset_points="") -> None:
        rows.append({**context, "reset_points": reset_points, "planner": name, **score_routes(changed, routes),
                     "runtime_s": runtime, "evaluations": evaluations, "iterations": iterations})
        for t in trace:
            traces.append({**{k: context[k] for k in ("phase", "scenario", "trial", "target_fraction", "locality")},
                           "planner": name, "evaluations": t[0], "elapsed_s": t[1], "best_objective": t[2],
                           "best_margin": t[3], "best_total_overdue": t[4]})

    if "greedy" in cfg.planners:
        t0 = time.perf_counter()
        routes = GreedyRouter()(changed)
        record("greedy", routes, time.perf_counter() - t0, 1, 0,
               [(1, time.perf_counter() - t0, *_obj(changed, routes))])

    if "vns" in cfg.planners:
        vns = VNSRouter(cfg.vns_params(), seed=replan_seed)
        res = vns.plan(changed)
        record("vns", res.routes, res.runtime_s, res.evaluations, res.iterations,
               [(t.iteration, t.elapsed_s, t.best_objective, t.best_margin, t.best_total_overdue) for t in res.trace])

    for strategy in STRATEGY_ORDER:
        if f"aco_{strategy.value}" not in cfg.planners:
            continue
        router = ACORouter(cfg.aco_params(), strategy=strategy, seed=replan_seed)
        router.restore(snap)
        router.reseed(replan_seed)
        result = router.plan(inst1, change)
        acted = router.last_change
        reset = len(acted.reset_scope) if strategy is StrategyName.SELECTIVE_RESET and acted is not None else ""
        record(f"aco_{strategy.value}", result.plan.for_drones(inst1.drone_ids), result.runtime_s,
               result.iterations * cfg.ants, result.iterations,
               [(t.iteration * cfg.ants, t.elapsed_s, t.best_objective, t.best_margin, t.best_total_overdue)
                for t in result.trace], reset)
    return rows, traces


def _obj(state: PlanningState, routes: dict) -> tuple[float, float, float]:
    s = score_routes(state, routes)
    return s["objective"], s["margin"], s["total_overdue"]


def _trial_job(args):
    cfg, seed, fraction, locality = args
    base = synthetic_state(cfg.n_points, cfg.drones, seed, cell_size_m=cfg.cell_size_m,
                           sensor_radius_m=cfg.sensor_radius_m)
    return run_pair(base, ChangeSpec(fraction, locality, cfg.kind, cfg.direction), cfg, seed)


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


def _state_job(args):
    cfg, label, base, seed, fraction, locality = args
    return run_pair(base, ChangeSpec(fraction, locality, cfg.kind, cfg.direction), cfg, seed, label=label)


def run_from_states(bases: dict[str, PlanningState], cfg: ExperimentConfig, progress: Callable[[str], None] = print):
    jobs = [(cfg, label, base, seed, f, loc) for label, base in bases.items()
            for seed in range(cfg.first_seed, cfg.first_seed + cfg.trials)
            for f, loc in product(cfg.fractions, cfg.localities)]
    rows: list[dict] = []
    traces: list[dict] = []
    if cfg.workers > 1:
        with ProcessPoolExecutor(cfg.workers) as pool:
            outputs = pool.map(_state_job, jobs)
            for job, (r, t) in zip(jobs, outputs):
                rows += r
                traces += t
                progress(f"{job[1]} seed {job[3]} fraction {job[4]:.2f} {job[5]} done")
    else:
        for job in jobs:
            r, t = _state_job(job)
            rows += r
            traces += t
            progress(f"{job[1]} seed {job[3]} fraction {job[4]:.2f} {job[5]} done")
    return rows, traces


def write_csv(path: str, fields: list[str], rows: Iterable[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
