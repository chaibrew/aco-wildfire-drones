from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import stats

BatchEvaluate = Callable[[Sequence[int], int], Sequence[object]]
BlockScores = Callable[[Sequence[object]], Sequence[float]]


@dataclass
class RaceStep:
    instance: int
    alive_before: int
    alive_after: int
    test: str
    p_value: float
    eliminated: tuple[int, ...]


@dataclass
class RaceResult:
    best: int
    alive: list[int]
    costs: np.ndarray
    instances_used: int
    eliminated_at: dict[int, int] = field(default_factory=dict)
    steps: list[RaceStep] = field(default_factory=list)
    raw: list[dict[int, object]] = field(default_factory=list)

    def mean_rank(self, candidates: Sequence[int] | None = None) -> dict[int, float]:
        cand = list(self.alive if candidates is None else candidates)
        block = self.costs[: self.instances_used][:, cand]
        ranks = np.apply_along_axis(stats.rankdata, 1, block)
        return dict(zip(cand, ranks.mean(axis=0)))


def friedman_survivors(block: np.ndarray, alpha: float) -> tuple[list[int], float, str]:
    n, k = block.shape
    if k == 2:
        diff = block[:, 0] - block[:, 1]
        if np.all(diff == 0):
            return [0, 1], 1.0, "wilcoxon"
        p = float(stats.wilcoxon(block[:, 0], block[:, 1], zero_method="zsplit").pvalue)
        if p >= alpha:
            return [0, 1], p, "wilcoxon"
        return ([0] if diff.mean() < 0 else [1]), p, "wilcoxon"
    ranks = np.apply_along_axis(stats.rankdata, 1, block)
    rank_sums = ranks.sum(axis=0)
    a = float((ranks**2).sum())
    c = n * k * (k + 1) ** 2 / 4.0
    if a - c <= 0:
        return list(range(k)), 1.0, "friedman"
    t = (k - 1) * float(((rank_sums - n * (k + 1) / 2.0) ** 2).sum()) / (a - c)
    p = float(1.0 - stats.chi2.cdf(t, k - 1))
    if p >= alpha:
        return list(range(k)), p, "friedman"
    df = (n - 1) * (k - 1)
    threshold = stats.t.ppf(1 - alpha / 2, df) * np.sqrt(2 * (n * a - float((rank_sums**2).sum())) / df)
    best = rank_sums.min()
    return [j for j in range(k) if rank_sums[j] - best <= threshold], p, "friedman"


def race(n_candidates: int, n_instances: int, evaluate: BatchEvaluate, *, first_test: int = 5,
         alpha: float = 0.05, block_scores: BlockScores | None = None,
         progress: Callable[[str], None] | None = None,
         stop: Callable[[Sequence[int], int], bool] | None = None) -> RaceResult:
    if n_candidates < 1 or n_instances < 1:
        raise ValueError("need at least one candidate and one instance")
    score = block_scores or (lambda results: [float(r) for r in results])
    results: list[dict[int, object]] = []
    costs = np.full((n_instances, n_candidates), np.nan)
    alive = list(range(n_candidates))
    eliminated: dict[int, int] = {}
    steps: list[RaceStep] = []
    used = 0
    for inst in range(n_instances):
        if stop is not None and inst >= first_test and stop(alive, inst):
            break
        results.append(dict(zip(alive, evaluate(alive, inst))))
        used = inst + 1
        block = np.array([score([results[i][c] for c in alive]) for i in range(used)], dtype=float)
        costs[:used] = np.nan
        costs[:used][:, alive] = block
        p, test, dropped = float("nan"), "none", []
        before = len(alive)
        if used >= first_test and len(alive) > 1:
            keep, p, test = friedman_survivors(costs[:used][:, alive], alpha)
            dropped = [cand for j, cand in enumerate(alive) if j not in keep]
            for cand in dropped:
                eliminated[cand] = used
            alive = [alive[j] for j in keep]
            if dropped:  # re-rank earlier blocks without the dropped candidates
                block = np.array([score([results[i][c] for c in alive]) for i in range(used)], dtype=float)
                costs[:used] = np.nan
                costs[:used][:, alive] = block
        steps.append(RaceStep(used, before, len(alive), test, p, tuple(dropped)))
        if progress:
            progress(f"instance {used}/{n_instances}: {len(alive)} alive"
                     + (f" ({test} p={p:.3g})" if test != "none" else ""))
        if len(alive) == 1:
            break
    result = RaceResult(alive[0], alive, costs, used, eliminated, steps, results)
    ranks = result.mean_rank()
    result.best = min(alive, key=lambda c: (ranks[c], np.nanmean(costs[:used, c])))
    return result
