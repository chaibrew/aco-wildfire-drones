from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import stats
from scipy.stats import qmc

BlockScores = Callable[[Sequence[object]], Sequence[float]]
Evaluate = Callable[[Sequence[int], int], Sequence[object]]
Progress = Callable[[str], None]


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    log: bool = False

    def from_unit(self, x: float) -> float:
        if self.log:
            return math.exp(math.log(self.low) + x * (math.log(self.high) - math.log(self.low)))
        return self.low + x * (self.high - self.low)

    def to_unit(self, v: float) -> float:
        if self.log:
            x = (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        else:
            x = (v - self.low) / (self.high - self.low)
        return min(1.0, max(0.0, x))


@dataclass(frozen=True)
class ParameterSpace:
    parameters: tuple[Parameter, ...]
    decimals: int = 4

    def __len__(self) -> int:
        return len(self.parameters)

    def __iter__(self):
        return iter(self.parameters)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parameters)

    def from_unit(self, u: Sequence[float]) -> dict[str, float]:
        return {p.name: round(float(p.from_unit(x)), self.decimals) for x, p in zip(u, self.parameters)}

    def to_unit(self, values: dict[str, float]) -> np.ndarray:
        return np.array([p.to_unit(values[p.name]) for p in self.parameters])

    def sobol(self, n: int, seed: int) -> list[dict[str, float]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sample = qmc.Sobol(d=len(self), scramble=True, seed=seed).random(n)
        return [self.from_unit(u) for u in sample]


@dataclass(frozen=True)
class Elimination:
    survivors: list[int]
    p_value: float
    test: str


def wilcoxon_test(block: np.ndarray, alpha: float) -> Elimination:
    diff = block[:, 0] - block[:, 1]
    if np.all(diff == 0):
        return Elimination([0, 1], 1.0, "wilcoxon")
    p = float(stats.wilcoxon(block[:, 0], block[:, 1], zero_method="zsplit").pvalue)
    if p >= alpha:
        return Elimination([0, 1], p, "wilcoxon")
    return Elimination([0] if diff.mean() < 0 else [1], p, "wilcoxon")


def friedman_conover_test(block: np.ndarray, alpha: float) -> Elimination:
    n, k = block.shape
    everyone = list(range(k))
    ranks = np.apply_along_axis(stats.rankdata, 1, block)
    rank_sums = ranks.sum(axis=0)
    a = float((ranks**2).sum())
    c = n * k * (k + 1) ** 2 / 4.0
    if a - c <= 0:
        return Elimination(everyone, 1.0, "friedman")
    t = (k - 1) * float(((rank_sums - n * (k + 1) / 2.0) ** 2).sum()) / (a - c)
    p = float(1.0 - stats.chi2.cdf(t, k - 1))
    if p >= alpha:
        return Elimination(everyone, p, "friedman")
    df = (n - 1) * (k - 1)
    threshold = stats.t.ppf(1 - alpha / 2, df) * np.sqrt(2 * (n * a - float((rank_sums**2).sum())) / df)
    best = rank_sums.min()
    return Elimination([j for j in everyone if rank_sums[j] - best <= threshold], p, "friedman")


def elimination_test(block: np.ndarray, alpha: float) -> Elimination:
    return (wilcoxon_test if block.shape[1] == 2 else friedman_conover_test)(block, alpha)


@dataclass(frozen=True)
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

    def mean_rank(self, candidates: Sequence[int] | None = None) -> dict[int, float]:
        cand = list(self.alive if candidates is None else candidates)
        ranks = np.apply_along_axis(stats.rankdata, 1, self.costs[: self.instances_used][:, cand])
        return dict(zip(cand, ranks.mean(axis=0)))


def race(n_candidates: int, n_instances: int, evaluate: Evaluate, *, first_test: int = 5, alpha: float = 0.05,
         block_scores: BlockScores | None = None, progress: Progress | None = None,
         stop: Callable[[Sequence[int], int], bool] | None = None) -> RaceResult:
    if n_candidates < 1 or n_instances < 1:
        raise ValueError("need at least one candidate and one instance")
    score = block_scores or (lambda results: [float(r) for r in results])
    results: list[dict[int, object]] = []
    costs = np.full((n_instances, n_candidates), np.nan)
    alive = list(range(n_candidates))
    eliminated: dict[int, int] = {}
    steps: list[RaceStep] = []

    def rescore() -> None:
        # every earlier block is re-ranked on the candidates still alive
        used = len(results)
        costs[:used] = np.nan
        costs[:used][:, alive] = np.array([score([r[c] for c in alive]) for r in results], dtype=float)

    for inst in range(n_instances):
        if stop is not None and inst >= first_test and stop(alive, inst):
            break
        results.append(dict(zip(alive, evaluate(alive, inst))))
        used = len(results)
        rescore()
        before, outcome, dropped = len(alive), Elimination(alive, float("nan"), "none"), []
        if used >= first_test and len(alive) > 1:
            outcome = elimination_test(costs[:used][:, alive], alpha)
            dropped = [c for j, c in enumerate(alive) if j not in outcome.survivors]
            eliminated.update({c: used for c in dropped})
            alive = [alive[j] for j in outcome.survivors]
            if dropped:
                rescore()
        steps.append(RaceStep(used, before, len(alive), outcome.test, outcome.p_value, tuple(dropped)))
        if progress:
            progress(f"instance {used}/{n_instances}: {len(alive)} alive"
                     + (f" ({outcome.test} p={outcome.p_value:.3g})" if outcome.test != "none" else ""))
        if len(alive) == 1:
            break
    used = len(results)
    result = RaceResult(alive[0], alive, costs, used, eliminated, steps)
    ranks = result.mean_rank()
    result.best = min(alive, key=lambda c: (ranks[c], np.nanmean(costs[:used, c])))
    return result


@dataclass(frozen=True)
class Candidate:
    id: int
    values: dict[str, float]
    iteration: int
    parent: int | None = None


CandidateEvaluate = Callable[[Sequence[Candidate], int], Sequence[object]]


class CachedEvaluator:
    def __init__(self, evaluate: CandidateEvaluate):
        self._evaluate = evaluate
        self.cache: dict[tuple[int, int], object] = {}
        self.experiments = 0

    def missing(self, candidates: Sequence[Candidate], inst: int) -> list[Candidate]:
        return [c for c in candidates if (c.id, inst) not in self.cache]

    def __call__(self, candidates: Sequence[Candidate], inst: int) -> list[object]:
        todo = self.missing(candidates, inst)
        if todo:
            for c, r in zip(todo, self._evaluate(todo, inst)):
                self.cache[(c.id, inst)] = r
            self.experiments += len(todo)
        return [self.cache[(c.id, inst)] for c in candidates]

    def instances_seen(self, candidate_id: int) -> int:
        return sum(1 for cid, _ in self.cache if cid == candidate_id)


@dataclass
class IraceResult:
    best: int
    elites: list[int]
    candidates: list[Candidate]
    iterations: list[dict]
    experiments: int
    instances_seen: dict[int, int]


class IteratedRace:
    def __init__(self, space: ParameterSpace, evaluate: CandidateEvaluate, n_instances: int, *, budget: int,
                 initial: Sequence[dict] = (), seed: int = 0, first_test: int = 5, alpha: float = 0.05,
                 block_scores: BlockScores | None = None, n_iterations: int | None = None,
                 progress: Progress | None = None):
        d = len(space)
        self.space = space
        self.evaluator = CachedEvaluator(evaluate)
        self.n_instances = n_instances
        self.budget = budget
        self.initial = list(initial)
        self.seed = seed
        self.first_test = first_test
        self.alpha = alpha
        self.block_scores = block_scores
        self.n_iterations = n_iterations or 2 + int(math.floor(math.log2(d)))
        self.n_elites = 2 + int(math.floor(math.log2(d)))
        self.progress = progress or (lambda _m: None)
        self.rng = np.random.default_rng(seed)
        self.sd = np.full(d, 0.5)
        self.candidates: list[Candidate] = []
        self.elites: list[int] = []
        self.elite_rank: dict[int, float] = {}
        self.iterations: list[dict] = []

    def run(self) -> IraceResult:
        for it in range(1, self.n_iterations + 1):
            remaining = self.budget - self.evaluator.experiments
            if remaining <= 0:
                break
            self._iteration(it, remaining)
        return IraceResult(self.elites[0], self.elites, self.candidates, self.iterations,
                           self.evaluator.experiments,
                           {c.id: self.evaluator.instances_seen(c.id) for c in self.candidates})

    def _iteration(self, it: int, remaining: int) -> None:
        b_j = remaining // (self.n_iterations - it + 1)
        n_j = max(len(self.elites) + 1, b_j // (self.first_test + min(5, it)))
        n_new = max(1, n_j - len(self.elites))
        new = self._sample_initial(n_new) if it == 1 else self._sample_near_elites(it, n_new)
        racers = [self.candidates[i] for i in self.elites] + new
        before = self.evaluator.experiments
        r = self._race(racers)
        ranks = r.mean_rank()
        survivors = sorted(r.alive, key=lambda j: ranks[j])[: self.n_elites]
        self.elites = [racers[j].id for j in survivors]
        self.elite_rank = {racers[j].id: ranks[j] for j in survivors}
        self.iterations.append({
            "iteration": it, "candidates": len(racers), "new": len(new), "instances": r.instances_used,
            "survivors": len(r.alive), "elites": " ".join(map(str, self.elites)),
            "experiments": self.evaluator.experiments - before, "sd": " ".join(f"{x:.3f}" for x in self.sd),
        })
        self.progress(f"iteration {it}: {len(racers)} candidates ({len(new)} new), {r.instances_used} instances, "
                      f"{len(r.alive)} survived, elites {self.elites}, "
                      f"{self.evaluator.experiments}/{self.budget} experiments")

    def _add(self, values: dict, iteration: int, parent: int | None = None) -> Candidate:
        c = Candidate(len(self.candidates), dict(values), iteration, parent)
        self.candidates.append(c)
        return c

    def _sample_initial(self, n_new: int) -> list[Candidate]:
        new = [self._add(v, 1) for v in self.initial]
        if n_new > len(new):
            new += [self._add(v, 1) for v in self.space.sobol(n_new - len(new), self.seed)]
        return new

    def _sample_near_elites(self, it: int, n_new: int) -> list[Candidate]:
        self.sd = self.sd * (1.0 / n_new) ** (1.0 / len(self.space))
        ordered = sorted(self.elites, key=lambda c: self.elite_rank[c])
        w = np.array([len(ordered) - r for r in range(len(ordered))], dtype=float)
        new = []
        for _ in range(n_new):
            parent = ordered[self.rng.choice(len(ordered), p=w / w.sum())]
            centre = self.space.to_unit(self.candidates[parent].values)
            u = [_truncnorm(self.rng, centre[k], self.sd[k]) for k in range(len(self.space))]
            new.append(self._add(self.space.from_unit(u), it, parent))
        return new

    def _race(self, racers: list[Candidate]) -> RaceResult:
        def evaluate(alive: Sequence[int], inst: int) -> list[object]:
            return self.evaluator([racers[j] for j in alive], inst)

        def over_budget(alive: Sequence[int], inst: int) -> bool:
            cost = len(self.evaluator.missing([racers[j] for j in alive], inst))
            return self.evaluator.experiments + cost > self.budget

        return race(len(racers), self.n_instances, evaluate, first_test=self.first_test, alpha=self.alpha,
                    block_scores=self.block_scores, stop=over_budget)


def iterated_race(space: ParameterSpace, evaluate: CandidateEvaluate, n_instances: int, **options) -> IraceResult:
    return IteratedRace(space, evaluate, n_instances, **options).run()


def _truncnorm(rng: np.random.Generator, mean: float, sd: float) -> float:
    for _ in range(1000):
        x = rng.normal(mean, sd)
        if 0.0 <= x <= 1.0:
            return float(x)
    return float(min(1.0, max(0.0, mean)))
