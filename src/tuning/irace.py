"""A small iterated race in the style of irace (Lopez-Ibanez et al., 2016), in Python.

Plain F-Race races one fixed sample of candidates. irace repeats the race: after each race the
best survivors ("elites") are kept, new candidates are sampled near them, and the sampling
spread shrinks from one iteration to the next, so later races search a smaller region around
the settings that did well.

What follows irace's defaults:
  * number of iterations N_iter = 2 + floor(log2(d)) for d parameters;
  * the remaining budget is split evenly over the remaining iterations, and iteration j races
    floor(B_j / (mu + min(5, j))) candidates (mu = first_test = 5);
  * up to N_min = 2 + floor(log2(d)) elites are kept (best mean rank among the survivors);
  * a new candidate picks an elite as its parent with probability proportional to
    (N_elites - rank + 1), and each parameter is drawn from a normal distribution centred on the
    parent's value (on the unit scale, truncated to [0, 1]); the standard deviation starts at 0.5
    and is multiplied by (1 / N_new)^(1 / d) every iteration.

Simplifications (stated so the report can say what was run):
  * every race uses the same fixed list of instances, in the same order, and results already
    computed for a (candidate, instance) pair are reused instead of rerun;
  * elites can be dropped by a race like any other candidate (irace's "elitist" rule, which
    protects elites until new instances have been seen, is not implemented);
  * only numerical parameters are sampled.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.stats import qmc

from .frace import BlockScores, race

# evaluate(list of parameter-value dicts, instance index) -> one raw result per dict
ConfigEvaluate = Callable[[Sequence[dict], int], Sequence[object]]


@dataclass
class IraceResult:
    best: int  # id of the best final elite
    elites: list[int]
    configs: list[dict]  # id -> parameter values (unit-cube point under "_unit")
    parent: list[int | None]  # id -> parent id (None for the first iteration and initial settings)
    born: list[int]  # id -> iteration in which it was sampled
    iterations: list[dict] = field(default_factory=list)  # one summary row per iteration
    experiments: int = 0  # (candidate, instance) evaluations actually run
    cache: dict[tuple[int, int], object] = field(default_factory=dict)


def iterated_race(space: Sequence[tuple[str, float, float, bool]],
                  to_values: Callable[[np.ndarray], dict], to_unit: Callable[[dict], np.ndarray],
                  evaluate: ConfigEvaluate, n_instances: int, *, budget: int, initial: Sequence[dict] = (),
                  seed: int = 0, first_test: int = 5, alpha: float = 0.05,
                  block_scores: BlockScores | None = None, n_iterations: int | None = None,
                  progress: Callable[[str], None] | None = None) -> IraceResult:
    d = len(space)
    rng = np.random.default_rng(seed)
    n_iter = n_iterations or 2 + int(math.floor(math.log2(d)))
    n_min = 2 + int(math.floor(math.log2(d)))
    mu = first_test
    res = IraceResult(best=-1, elites=[], configs=[], parent=[], born=[])
    say = progress or (lambda _m: None)

    def add(values: dict, parent: int | None, it: int) -> int:
        res.configs.append(values)
        res.parent.append(parent)
        res.born.append(it)
        return len(res.configs) - 1

    sd = np.full(d, 0.5)
    elites: list[int] = []
    elite_rank: dict[int, float] = {}
    for it in range(1, n_iter + 1):
        remaining = budget - res.experiments
        if remaining <= 0:
            break
        b_j = remaining // (n_iter - it + 1)
        n_j = max(len(elites) + 1, b_j // (mu + min(5, it)))
        n_new = max(1, n_j - len(elites))
        new: list[int] = []
        if it == 1:
            for v in initial:
                new.append(add(dict(v), None, it))
            if n_new > len(new):
                with warnings.catch_warnings():  # balance warning for n not a power of 2: harmless here
                    warnings.simplefilter("ignore", UserWarning)
                    sample = qmc.Sobol(d=d, scramble=True, seed=seed).random(n_new - len(new))
                new += [add(to_values(u), None, it) for u in sample]
        else:
            sd = sd * (1.0 / n_new) ** (1.0 / d)
            ordered = sorted(elites, key=lambda c: elite_rank[c])
            w = np.array([len(ordered) - r for r in range(len(ordered))], dtype=float)
            for _ in range(n_new):
                parent = ordered[rng.choice(len(ordered), p=w / w.sum())]
                centre = to_unit(res.configs[parent])
                u = np.array([_truncnorm(rng, centre[k], sd[k]) for k in range(d)])
                new.append(add(to_values(u), parent, it))
        cands = elites + new

        def batch(alive_local: Sequence[int], inst: int) -> list[object]:
            ids = [cands[j] for j in alive_local]
            todo = [c for c in ids if (c, inst) not in res.cache]
            if todo:
                out = evaluate([res.configs[c] for c in todo], inst)
                for c, r in zip(todo, out):
                    res.cache[(c, inst)] = r
                res.experiments += len(todo)
            return [res.cache[(c, inst)] for c in ids]

        def stop(alive_local: Sequence[int], inst: int) -> bool:
            cost = sum((cands[j], inst) not in res.cache for j in alive_local)
            return res.experiments + cost > budget

        used_before = res.experiments
        r = race(len(cands), n_instances, batch, first_test=first_test, alpha=alpha,
                 block_scores=block_scores, stop=stop)
        ranks = r.mean_rank()
        survivors = sorted(r.alive, key=lambda j: ranks[j])[:n_min]
        elites = [cands[j] for j in survivors]
        elite_rank = {cands[j]: ranks[j] for j in survivors}
        res.iterations.append({
            "iteration": it, "candidates": len(cands), "new": len(new), "instances": r.instances_used,
            "survivors": len(r.alive), "elites": " ".join(map(str, elites)),
            "experiments": res.experiments - used_before, "sd": " ".join(f"{x:.3f}" for x in sd),
        })
        say(f"iteration {it}: {len(cands)} candidates ({len(new)} new), {r.instances_used} instances, "
            f"{len(r.alive)} survived, elites {elites}, {res.experiments}/{budget} experiments")
    res.elites = elites
    res.best = elites[0]
    return res


def _truncnorm(rng: np.random.Generator, mean: float, sd: float) -> float:
    for _ in range(1000):
        x = rng.normal(mean, sd)
        if 0.0 <= x <= 1.0:
            return float(x)
    return float(min(1.0, max(0.0, mean)))
