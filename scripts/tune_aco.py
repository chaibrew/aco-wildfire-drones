"""ACO parameter tuning: budget check, F-Race over the shared parameters, an iterated race (irace-style)
over the same space, and F-Race over fast evaporation's own parameters.

Every step uses its own instance seeds and Camp Fire states (tuning.instances.SEEDS), none of which
overlap the reported experiments (seeds from 0; Camp Fire fire seed 0, stages 6/40/80).

Examples
  python scripts/tune_aco.py budget --camp --instances 20 --workers 8 --out outputs/tuning/budget
  python scripts/tune_aco.py shared --camp --ants 20 --iterations 60 --instances 30 --workers 8 \
      --out outputs/tuning/shared
  python scripts/tune_aco.py irace --camp --ants 20 --iterations 60 --instances 30 --budget 1000 \
      --workers 8 --out outputs/tuning/irace
  python scripts/tune_aco.py fast-evaporation --params outputs/tuning/chosen_shared.json --camp \
      --ants 20 --iterations 60 --instances 30 --workers 8 --out outputs/tuning/fast_evaporation
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from aco_router import StrategyName  # noqa: E402
from tuning import (SEEDS, SHARED_NAMES, SHARED_SPACE, UNTUNED_PARAMS, block_scores, camp_states, fast_evaporation_grid,  # noqa: E402
                    from_unit, iterated_race, load_params, make_instances, race, save_params, sobol_candidates,
                    to_unit)
from tuning.run import (ParallelEvaluator, budget_summary, race_candidate_rows, race_log_rows,  # noqa: E402
                        result_rows, write_csv)

DEFAULT_BUDGETS = ("5x10", "10x15", "10x30", "20x30", "20x60", "40x60")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="step", required=True)
    for name in ("budget", "shared", "irace", "fast-evaporation"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, help="output prefix, e.g. outputs/tuning/shared")
        s.add_argument("--params", default=None, help="JSON of base parameters (default: the untuned parameters)")
        s.add_argument("--camp", action="store_true", help="alternate synthetic and captured Camp Fire states")
        s.add_argument("--instances", type=int, default=30)
        s.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        s.add_argument("--warmup", type=int, default=5)
        s.add_argument("--drones", type=int, default=6)
        s.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
        s.add_argument("--cache", default="data/cache")
        if name == "budget":
            s.add_argument("--budgets", nargs="+", default=list(DEFAULT_BUDGETS), help="ANTSxITERATIONS")
        else:
            s.add_argument("--ants", type=int, default=20)
            s.add_argument("--iterations", type=int, default=60)
            s.add_argument("--first-test", type=int, default=5)
            s.add_argument("--alpha", type=float, default=0.05)
        if name == "shared":
            s.add_argument("--candidates", type=int, default=32, help="Sobol settings besides the defaults")
            s.add_argument("--sobol-seed", type=int, default=0)
        if name == "irace":
            s.add_argument("--budget", type=int, default=1000, help="(candidate, instance) evaluations")
            s.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup(a, step: str):
    base = load_params(a.params) if a.params else UNTUNED_PARAMS
    seeds = SEEDS[step]
    camp = camp_states(seeds, gpkg=a.gpkg, cache=a.cache, drones=a.drones) if a.camp else None
    instances = make_instances(seeds, a.instances, camp, drones=a.drones)
    print(f"{step}: {len(instances)} instances, seeds {seeds.first_seed}-{seeds.first_seed + len(instances) - 1}"
          + (f", Camp Fire fire seed {seeds.fire_seed} stages {seeds.stages}" if camp else ""), flush=True)
    return base, instances


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_budget(a):
    base, instances = setup(a, "budget")
    budgets = [tuple(int(x) for x in b.lower().split("x")) for b in a.budgets]
    rows = []
    with ParallelEvaluator(instances, a.workers, warmup=a.warmup) as ev:
        for ants, iters in budgets:
            results = ev.across_instances(base.with_(n_ants=ants, max_iterations=iters))
            for i, res in enumerate(results):
                rows += result_rows(instances, i, f"{ants}x{iters}", res, ants=ants, iterations=iters)
            log(f"budget {ants}x{iters} done")
    write_csv(f"{a.out}.csv", rows)
    summary = budget_summary(rows, budgets)
    write_csv(f"{a.out}_summary.csv", summary)
    print(f"\n{'budget':>7s} {'evals':>6s} {'margin':>7s} {'tot.over':>8s} {'n.over':>7s} {'s/plan':>7s}"
          f" {'next b/t/w':>11s} {'p(next)':>8s}")
    for r in summary:
        nxt = f"{r['next_better']}/{r['next_tied']}/{r['next_worse']}" if "next" in r else "-"
        p = f"{r['p_total_overdue_vs_next']:.3g}" if "next" in r else "-"
        print(f"{r['budget']:>7s} {r['evaluations']:6d} {r['margin']:7.3f} {r['total_overdue']:8.2f} "
              f"{r['overdue_count']:7.2f} {r['seconds_per_plan']:7.2f} {nxt:>11s} {p:>8s}")
    print(f"\nwrote {a.out}.csv and {a.out}_summary.csv")


def write_race(a, candidates, labels, result, rows, names, meta):
    write_csv(f"{a.out}_candidates.csv", race_candidate_rows(result, candidates, names, labels))
    write_csv(f"{a.out}_log.csv", race_log_rows(result))
    write_csv(f"{a.out}_results.csv", rows)
    save_params(f"{a.out}_params.json", candidates[result.best], candidate=result.best,
                survivors=result.alive, instances_raced=result.instances_used, **meta)
    for c in result.alive:
        save_params(f"{a.out}_candidate{c}.json", candidates[c], candidate=c, **meta)
    print(f"survivors: {result.alive}; race pick: candidate {result.best}")
    for c in result.alive:
        print(f"  candidate {c}: " + ", ".join(f"{n}={getattr(candidates[c], n)}" for n in names))
    print(f"wrote {a.out}_candidates.csv, _log.csv, _results.csv, _params.json, _candidate<N>.json")


def run_race(a, step, candidates, labels, strategies, names, meta):
    _, instances = setup(a, step)
    rows: list[dict] = []
    with ParallelEvaluator(instances, a.workers, strategies, a.warmup) as ev:
        def evaluate(alive, inst):
            out = ev([candidates[c] for c in alive], inst)
            for c, res in zip(alive, out):
                rows.extend(result_rows(instances, inst, c, res))
            return out
        result = race(len(candidates), len(instances), evaluate, first_test=a.first_test, alpha=a.alpha,
                      block_scores=block_scores, progress=log)
    write_race(a, candidates, labels, result, rows, names, meta)
    return result


def run_shared(a):
    base = (load_params(a.params) if a.params else UNTUNED_PARAMS).with_(n_ants=a.ants, max_iterations=a.iterations)
    candidates = sobol_candidates(base, a.candidates, a.sobol_seed)
    run_race(a, "shared", candidates, {0: "current defaults"}, tuple(StrategyName), SHARED_NAMES,
             {"step": "shared", "ants": a.ants, "iterations": a.iterations, "race_alpha": a.alpha})


def run_fast_evaporation(a):
    if not a.params:
        sys.exit("fast-evaporation needs --params (the chosen shared setting)")
    base = load_params(a.params).with_(n_ants=a.ants, max_iterations=a.iterations)
    candidates = fast_evaporation_grid(base)
    print(f"{len(candidates)} candidates (rho = {base.rho}; rho_high values <= rho skipped)")
    result = run_race(a, "fast-evaporation", candidates, {0: "current"}, (StrategyName.FAST_EVAPORATION,),
                      ("rho_high", "fast_evaporation_iterations"),
                      {"step": "fast-evaporation", "ants": a.ants, "iterations": a.iterations, "race_alpha": a.alpha})
    if len(result.alive) == len(candidates):
        print("every candidate survived: no evidence to change the current rho_high and k")


def run_irace(a):
    base = (load_params(a.params) if a.params else UNTUNED_PARAMS).with_(n_ants=a.ants, max_iterations=a.iterations)
    _, instances = setup(a, "irace")
    rows: list[dict] = []
    defaults = {n: getattr(base, n) for n in SHARED_NAMES}
    ids_seen: dict[tuple, int] = {}
    with ParallelEvaluator(instances, a.workers, warmup=a.warmup) as ev:
        def evaluate(configs, inst):
            out = ev([base.with_(**c) for c in configs], inst)
            for c, res in zip(configs, out):
                key = tuple(c[n] for n in SHARED_NAMES)
                rows.extend(result_rows(instances, inst, ids_seen.setdefault(key, len(ids_seen)), res))
            return out
        res = iterated_race(SHARED_SPACE, from_unit, to_unit, evaluate, len(instances), budget=a.budget,
                            initial=[defaults], seed=a.seed, first_test=a.first_test, alpha=a.alpha,
                            block_scores=block_scores, progress=log)
    # result rows were labelled in evaluation order; relabel them with irace's own candidate ids
    by_key = {tuple(c[n] for n in SHARED_NAMES): i for i, c in enumerate(res.configs)}
    inv = {v: by_key[k] for k, v in ids_seen.items()}
    for r in rows:
        r["setting"] = inv[r["setting"]]
    cand_rows = []
    for i, c in enumerate(res.configs):
        n_inst = sum(1 for (cid, _inst) in res.cache if cid == i)
        cand_rows.append({"candidate": i, "iteration": res.born[i], "parent": "" if res.parent[i] is None else
                          res.parent[i], **c, "instances_evaluated": n_inst, "final_elite": int(i in res.elites),
                          "elite_rank": res.elites.index(i) + 1 if i in res.elites else "",
                          "label": "current defaults" if i == 0 else ""})
    write_csv(f"{a.out}_candidates.csv", cand_rows)
    write_csv(f"{a.out}_iterations.csv", res.iterations)
    write_csv(f"{a.out}_results.csv", rows)
    meta = {"step": "irace", "ants": a.ants, "iterations": a.iterations, "budget": a.budget,
            "experiments": res.experiments}
    save_params(f"{a.out}_params.json", base.with_(**res.configs[res.best]), candidate=res.best,
                elites=res.elites, **meta)
    for e in res.elites:
        save_params(f"{a.out}_elite{e}.json", base.with_(**res.configs[e]), candidate=e, **meta)
    print(f"final elites (best first): {res.elites}")
    for e in res.elites:
        print(f"  candidate {e}: " + ", ".join(f"{n}={res.configs[e][n]}" for n in SHARED_NAMES))
    print(f"wrote {a.out}_candidates.csv, _iterations.csv, _results.csv, _params.json, _elite<N>.json")


def main():
    a = parse_args()
    {"budget": run_budget, "shared": run_shared, "irace": run_irace,
     "fast-evaporation": run_fast_evaporation}[a.step](a)


if __name__ == "__main__":
    main()
