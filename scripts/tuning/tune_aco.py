from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from aco_router import ACOParams, StrategyName  # noqa: E402
from tuning import (SEEDS, SHARED_SPACE, UNTUNED_PARAMS, ParallelEvaluator, camp_states,  # noqa: E402
                    fast_evaporation_grid, iterated_race, load_params, make_instances, mean_strategy_rank, race,
                    save_params, sobol_candidates)
from tuning.tuner import (FAST_EVAPORATION_NAMES, budget_summary, confirmation_summary,  # noqa: E402
                          race_candidate_rows, race_log_rows, result_rows, setting_means, write_csv)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Step:
    name = ""
    seeds = ""
    default_instances = 30
    strategies: tuple[StrategyName, ...] = tuple(StrategyName)

    def add_arguments(self, p: argparse.ArgumentParser) -> None:
        p.add_argument("--ants", type=int, default=20)
        p.add_argument("--iterations", type=int, default=60)

    def base(self, a) -> ACOParams:
        params = load_params(a.params) if a.params else UNTUNED_PARAMS
        return params.with_(n_ants=a.ants, max_iterations=a.iterations)

    def run(self, a, ev: ParallelEvaluator) -> None:
        raise NotImplementedError


class Budget(Step):
    name = seeds = "budget"
    default_instances = 20
    budgets = ("5x10", "10x15", "10x30", "20x30", "20x60", "40x60")

    def add_arguments(self, p):
        p.add_argument("--budgets", nargs="+", default=list(self.budgets), help="ANTSxITERATIONS")

    def run(self, a, ev):
        base = load_params(a.params) if a.params else UNTUNED_PARAMS
        budgets = [tuple(int(x) for x in b.lower().split("x")) for b in a.budgets]
        rows = []
        for ants, iters in budgets:
            for i, scores in enumerate(ev.across_instances(base.with_(n_ants=ants, max_iterations=iters))):
                rows += result_rows(ev.instances, i, f"{ants}x{iters}", scores, ants=ants, iterations=iters)
            log(f"budget {ants}x{iters} done")
        summary = budget_summary(rows, budgets)
        write_csv(f"{a.out}.csv", rows)
        write_csv(f"{a.out}_summary.csv", summary)
        print(f"\n{'budget':>7s} {'evals':>6s} {'margin':>7s} {'tot.over':>8s} {'n.over':>7s} {'s/plan':>7s}"
              f" {'next b/t/w':>11s} {'p(next)':>8s}")
        for r in summary:
            nxt = f"{r['next_better']}/{r['next_tied']}/{r['next_worse']}" if "next" in r else "-"
            p = f"{r['p_total_overdue_vs_next']:.3g}" if "next" in r else "-"
            print(f"{r['budget']:>7s} {r['evaluations']:6d} {r['margin']:7.3f} {r['total_overdue']:8.2f} "
                  f"{r['overdue_count']:7.2f} {r['seconds_per_plan']:7.2f} {nxt:>11s} {p:>8s}")


class RacingStep(Step):
    names: tuple[str, ...] = ()
    labels: dict[int, str] = {}

    def add_arguments(self, p):
        super().add_arguments(p)
        p.add_argument("--first-test", type=int, default=5)
        p.add_argument("--alpha", type=float, default=0.05)

    def candidates(self, a) -> list[ACOParams]:
        raise NotImplementedError

    def run(self, a, ev):
        candidates = self.candidates(a)
        rows: list[dict] = []

        def evaluate(alive, inst):
            out = ev.on_instance([candidates[c] for c in alive], inst)
            for c, scores in zip(alive, out):
                rows.extend(result_rows(ev.instances, inst, c, scores))
            return out

        result = race(len(candidates), len(ev.instances), evaluate, first_test=a.first_test, alpha=a.alpha,
                      block_scores=mean_strategy_rank, progress=log)
        meta = {"step": self.name, "ants": a.ants, "iterations": a.iterations, "race_alpha": a.alpha}
        write_csv(f"{a.out}_candidates.csv", race_candidate_rows(result, candidates, self.names, self.labels))
        write_csv(f"{a.out}_log.csv", race_log_rows(result))
        write_csv(f"{a.out}_results.csv", rows)
        save_params(f"{a.out}_params.json", candidates[result.best], candidate=result.best,
                    survivors=result.alive, instances_raced=result.instances_used, **meta)
        for c in result.alive:
            save_params(f"{a.out}_candidate{c}.json", candidates[c], candidate=c, **meta)
        print(f"survivors: {result.alive}; race pick: candidate {result.best}")
        for c in result.alive:
            print(f"  candidate {c}: " + ", ".join(f"{n}={getattr(candidates[c], n)}" for n in self.names))
        if len(result.alive) == len(candidates):
            print("every candidate survived: no evidence to change the current setting")


class Shared(RacingStep):
    name = seeds = "shared"
    names = SHARED_SPACE.names
    labels = {0: "current defaults"}

    def add_arguments(self, p):
        super().add_arguments(p)
        p.add_argument("--candidates", type=int, default=32, help="Sobol settings besides the defaults")
        p.add_argument("--sobol-seed", type=int, default=0)

    def candidates(self, a):
        return sobol_candidates(self.base(a), a.candidates, a.sobol_seed)


class FastEvaporation(RacingStep):
    name = seeds = "fast-evaporation"
    names = FAST_EVAPORATION_NAMES
    labels = {0: "current"}
    strategies = (StrategyName.FAST_EVAPORATION,)

    def candidates(self, a):
        if not a.params:
            sys.exit("fast-evaporation needs --params (the chosen shared setting)")
        grid = fast_evaporation_grid(self.base(a))
        print(f"{len(grid)} candidates (rho = {grid[0].rho}; rho_high values <= rho skipped)")
        return grid


class Irace(Step):
    name = seeds = "irace"

    def add_arguments(self, p):
        super().add_arguments(p)
        p.add_argument("--first-test", type=int, default=5)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument("--budget", type=int, default=1000, help="(candidate, instance) evaluations")
        p.add_argument("--seed", type=int, default=0)

    def run(self, a, ev):
        base = self.base(a)
        rows: list[dict] = []

        def evaluate(candidates, inst):
            out = ev.on_instance([base.with_(**c.values) for c in candidates], inst)
            for c, scores in zip(candidates, out):
                rows.extend(result_rows(ev.instances, inst, c.id, scores))
            return out

        res = iterated_race(SHARED_SPACE, evaluate, len(ev.instances), budget=a.budget,
                            initial=[{n: getattr(base, n) for n in SHARED_SPACE.names}], seed=a.seed,
                            first_test=a.first_test, alpha=a.alpha, block_scores=mean_strategy_rank, progress=log)
        write_csv(f"{a.out}_candidates.csv", [
            {"candidate": c.id, "iteration": c.iteration, "parent": "" if c.parent is None else c.parent,
             **c.values, "instances_evaluated": res.instances_seen[c.id], "final_elite": int(c.id in res.elites),
             "elite_rank": res.elites.index(c.id) + 1 if c.id in res.elites else "",
             "label": "current defaults" if c.id == 0 else ""} for c in res.candidates])
        write_csv(f"{a.out}_iterations.csv", res.iterations)
        write_csv(f"{a.out}_results.csv", rows)
        meta = {"step": self.name, "ants": a.ants, "iterations": a.iterations, "budget": a.budget,
                "experiments": res.experiments}
        save_params(f"{a.out}_params.json", base.with_(**res.candidates[res.best].values), candidate=res.best,
                    elites=res.elites, **meta)
        for e in res.elites:
            save_params(f"{a.out}_elite{e}.json", base.with_(**res.candidates[e].values), candidate=e, **meta)
        print(f"final elites (best first): {res.elites}")
        for e in res.elites:
            print(f"  candidate {e}: " + ", ".join(f"{n}={v}" for n, v in res.candidates[e].values.items()))


class Confirm(Step):
    name = seeds = "confirm"
    default_instances = 20

    def add_arguments(self, p):
        super().add_arguments(p)
        p.add_argument("settings", nargs="+", help="NAME=defaults or NAME=params.json; the first is the reference")

    def run(self, a, ev):
        settings = []
        for item in a.settings:
            name, _, spec = item.partition("=")
            params = UNTUNED_PARAMS if spec in ("", "defaults") else load_params(spec)
            settings.append((name, params.with_(n_ants=a.ants, max_iterations=a.iterations)))
        rows = []
        for name, params in settings:
            for i, scores in enumerate(ev.across_instances(params)):
                rows += result_rows(ev.instances, i, name, scores)
            log(f"{name} done")
        ref, others = settings[0][0], [n for n, _ in settings[1:]]
        summary, means = confirmation_summary(rows, ref, others), setting_means(rows)
        write_csv(f"{a.out}.csv", rows)
        write_csv(f"{a.out}_summary.csv", summary)
        write_csv(f"{a.out}_means.csv", means)
        print(f"\n{'setting':>16s} {'margin':>7s} {'tot.over':>8s} {'n.over':>7s}")
        for m in means:
            print(f"{m['setting']:>16s} {m['margin']:7.3f} {m['total_overdue']:8.2f} {m['overdue_count']:7.2f}")
        print(f"\nvs {ref} (better/tied/worse, mean change, Wilcoxon p):")
        for r in summary:
            if r["measure"] != "rank_key" and (r["strategy"] == "all" or r["measure"] == "total_overdue"):
                print(f"  {r['setting']:>12s} {r['strategy']:>17s} {r['measure']:>14s} "
                      f"{r['better']:3d}/{r['tied']:3d}/{r['worse']:3d}  {r['mean_change']:+9.3f}  p={r['p']:.2g}")


STEPS = {s.name: s for s in (Budget(), Shared(), Irace(), FastEvaporation(), Confirm())}


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="step", required=True)
    for step in STEPS.values():
        s = sub.add_parser(step.name)
        s.add_argument("--out", required=True, help="output prefix, e.g. outputs/tuning/shared")
        s.add_argument("--params", default=None, help="JSON of base parameters (default: the untuned parameters)")
        s.add_argument("--camp", action="store_true", help="alternate synthetic and captured Camp Fire states")
        s.add_argument("--instances", type=int, default=step.default_instances)
        s.add_argument("--workers", type=int, default=os.cpu_count() or 1)
        s.add_argument("--warmup", type=int, default=5)
        s.add_argument("--drones", type=int, default=6)
        s.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
        s.add_argument("--cache", default="data/cache")
        step.add_arguments(s)
    return p.parse_args()


def main():
    a = parse_args()
    step = STEPS[a.step]
    seeds = SEEDS[step.seeds]
    camp = camp_states(seeds, gpkg=a.gpkg, cache=a.cache, drones=a.drones) if a.camp else None
    instances = make_instances(seeds, a.instances, camp, drones=a.drones)
    print(f"{step.name}: {len(instances)} instances, seeds {seeds.first_seed}-{seeds.first_seed + len(instances) - 1}"
          + (f", Camp Fire fire seed {seeds.fire_seed} stages {seeds.stages}" if camp else ""), flush=True)
    with ParallelEvaluator(instances, a.workers, step.strategies, a.warmup) as ev:
        step.run(a, ev)


if __name__ == "__main__":
    main()
