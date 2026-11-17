"""Variable Neighbourhood Search (VNS) baseline.

Follows the scheme Bailon-Ruiz et al. (2022) use for wildfire UAV routing: start from
a greedy solution, repeatedly *shake* the incumbent in a neighbourhood of increasing
size k, descend with a randomised variable neighbourhood descent (VND), and accept the
result only if it is better (then k resets to 1).

Fairness with the ACO router
  * Same ordering of solutions: `Metrics.score = (margin, total_overdue, distance)`,
    computed by the same `aco_router.evaluate` the ACO router uses.
  * Same budget unit: the number of *distinct* full plan evaluations. ACO spends
    n_ants * max_iterations of them per replan; set `VNSParams.max_evaluations` to that
    number (the experiment runner does this). A wall-clock budget is also supported.
  * Feasibility checks and cache hits are free; they are not evaluations.

The search is stateless between replans (cold start from greedy every time), so VNS
neither benefits from nor is hurt by what changed. That is deliberate: it is the
"no memory" reference for the pheromone-handling variants.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

from fire_sim.simulation import PlanningState

from aco_router.evaluator import evaluate
from aco_router.fire_sim_adapter import state_to_instance
from aco_router.model import Metrics, Plan, TraceEntry

from .greedy import greedy_routes

Routes = dict[int, list]

_TOL = 1e-9


@dataclass(frozen=True)
class VNSParams:
    max_evaluations: int = 150
    time_budget_s: float | None = None  # if set, stop on whichever limit is hit first
    k_max: int = 3  # largest shake strength (number of random moves)
    ls_samples: int = 8  # sampled neighbours per VND neighbourhood before moving on
    patience: int = 100  # stop after this many iterations that found no new distinct plan (tiny search spaces)

    def __post_init__(self) -> None:
        if self.max_evaluations < 1:
            raise ValueError("max_evaluations must be >= 1")
        if self.k_max < 1 or self.ls_samples < 1 or self.patience < 1:
            raise ValueError("k_max, ls_samples and patience must be >= 1")


@dataclass(frozen=True)
class VNSResult:
    routes: Routes
    metrics: Metrics
    evaluations: int
    iterations: int
    runtime_s: float
    trace: tuple[TraceEntry, ...]  # iteration = evaluations used so far


# --------------------------------------------------------------------------- moves
# Each move copies the routes, edits the copy and returns it (None if not applicable).
# Feasibility (horizon) is checked by the caller.

def _copy(routes: Routes) -> Routes:
    return {d: list(r) for d, r in routes.items()}


def _nonempty(routes: Routes, minimum: int = 1) -> list:
    return [d for d, r in routes.items() if len(r) >= minimum]


def relocate_intra(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    cands = _nonempty(routes, 2)
    if not cands:
        return None
    new = _copy(routes)
    r = new[rng.choice(cands)]
    a = rng.randrange(len(r))
    item = r.pop(a)
    b = rng.randrange(len(r) + 1)
    if b == a:
        return None
    r.insert(b, item)
    return new


def relocate_inter(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    src = _nonempty(routes)
    if not src or len(routes) < 2:
        return None
    s = rng.choice(src)
    d = rng.choice([k for k in routes if k != s])
    new = _copy(routes)
    item = new[s].pop(rng.randrange(len(new[s])))
    new[d].insert(rng.randrange(len(new[d]) + 1), item)
    return new


def swap(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    slots = [(d, k) for d, r in routes.items() for k in range(len(r))]
    if len(slots) < 2:
        return None
    (d1, k1), (d2, k2) = rng.sample(slots, 2)
    new = _copy(routes)
    new[d1][k1], new[d2][k2] = new[d2][k2], new[d1][k1]
    return new


def two_opt(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    cands = _nonempty(routes, 3)
    if not cands:
        return None
    new = _copy(routes)
    d = rng.choice(cands)
    i, j = sorted(rng.sample(range(len(new[d])), 2))
    new[d][i : j + 1] = reversed(new[d][i : j + 1])
    return new


def replace_visit(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    """Swap a visited point for another point (often one nobody visits)."""
    cands = _nonempty(routes)
    if not cands or not ids:
        return None
    new = _copy(routes)
    d = rng.choice(cands)
    k = rng.randrange(len(new[d]))
    visited = {i for r in new.values() for i in r}
    unvisited = [i for i in ids if i not in visited]
    pool = unvisited if unvisited and rng.random() < 0.8 else ids
    choice = rng.choice(pool)
    if choice == new[d][k]:
        return None
    new[d][k] = choice
    return new


def insert_visit(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    """Add a visit if there is slack left in the horizon."""
    if not ids:
        return None
    new = _copy(routes)
    d = rng.choice(list(new))
    new[d].insert(rng.randrange(len(new[d]) + 1), rng.choice(ids))
    return new


def remove_visit(routes: Routes, rng: random.Random, ids=None) -> Routes | None:
    cands = _nonempty(routes)
    if not cands:
        return None
    new = _copy(routes)
    d = rng.choice(cands)
    new[d].pop(rng.randrange(len(new[d])))
    return new


LOCAL_SEARCH_MOVES = (relocate_intra, relocate_inter, swap, two_opt, replace_visit, insert_visit)
SHAKE_MOVES = (relocate_inter, swap, two_opt, replace_visit, remove_visit)


# -------------------------------------------------------------------------- search

class _Problem:
    """Feasibility test and a counted, cached evaluation for one PlanningState."""

    def __init__(self, state: PlanningState):
        self.state = state
        self.instance = state_to_instance(state)
        self.ids = sorted(state.points)
        self.limit = state.time_s + state.horizon_s + _TOL
        self.evaluations = 0
        self._cache: dict[tuple, Metrics] = {}

    def route_fits(self, drone, route) -> bool:
        s = self.state
        x, y, clock = drone.x, drone.y, s.time_s
        for i in route:
            p = s.points[i]
            clock += math.hypot(p.x - x, p.y - y) / s.speed_mps + s.service_s
            if clock > self.limit:
                return False
            x, y = p.x, p.y
        return True

    def feasible(self, routes: Routes) -> bool:
        return all(self.route_fits(d, routes[d.id]) for d in self.state.drones)

    def evaluate(self, routes: Routes) -> Metrics:
        key = tuple((d.id, tuple(routes[d.id])) for d in self.state.drones)
        m = self._cache.get(key)
        if m is None:
            m = evaluate(self.instance, Plan(routes), validate=False)
            self._cache[key] = m
            self.evaluations += 1
        return m


class VNSRouter:
    """Callable planner: `router(PlanningState) -> {drone_id: [point ids]}`."""

    def __init__(self, params: VNSParams = VNSParams(), seed: int = 0, name: str = "vns"):
        self.params = params
        self.name = name
        self._rng = random.Random(seed)
        self.results: list[VNSResult] = []

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def __call__(self, state: PlanningState) -> Routes:
        result = self.plan(state)
        self.results.append(result)
        return {d: list(r) for d, r in result.routes.items()}

    def plan(self, state: PlanningState) -> VNSResult:
        start = time.perf_counter()
        prob = _Problem(state)
        best = greedy_routes(state)
        best_m = prob.evaluate(best)
        trace = [TraceEntry(prob.evaluations, time.perf_counter() - start, best_m.objective, best_m.margin,
                            best_m.total_overdue)]
        iterations = 0
        if prob.ids:
            k = 1
            stale = 0  # consecutive iterations that produced no new distinct plan
            while not self._spent(prob, start) and stale < self.params.patience:
                iterations += 1
                before = prob.evaluations
                x = self._shake(prob, best, k)
                if x is None:  # no applicable shake
                    stale += 1
                    continue
                m = prob.evaluate(x)
                x, m = self._descend(prob, x, m, start)
                if m.score < best_m.score:
                    best, best_m, k = x, m, 1
                else:
                    k = k + 1 if k < self.params.k_max else 1
                stale = stale + 1 if prob.evaluations == before else 0
                trace.append(TraceEntry(prob.evaluations, time.perf_counter() - start, best_m.objective,
                                        best_m.margin, best_m.total_overdue))
        return VNSResult(best, best_m, prob.evaluations, iterations, time.perf_counter() - start, tuple(trace))

    # -- internals
    def _spent(self, prob: _Problem, start: float) -> bool:
        if prob.evaluations >= self.params.max_evaluations:
            return True
        t = self.params.time_budget_s
        return t is not None and time.perf_counter() - start >= t

    def _apply(self, prob: _Problem, routes: Routes, moves, tries: int = 6) -> Routes | None:
        for _ in range(tries):
            move = self._rng.choice(moves) if len(moves) > 1 else moves[0]
            cand = move(routes, self._rng, prob.ids)
            if cand is not None and prob.feasible(cand):
                return cand
        return None

    def _shake(self, prob: _Problem, routes: Routes, k: int) -> Routes | None:
        cur = routes
        changed = False
        for _ in range(k):
            nxt = self._apply(prob, cur, SHAKE_MOVES)
            if nxt is not None:
                cur, changed = nxt, True
        return cur if changed else None

    def _descend(self, prob: _Problem, cur: Routes, cur_m: Metrics, start: float):
        """Randomised VND: sample neighbours of N_1..N_6, first improvement, restart at N_1."""
        n = 0
        while n < len(LOCAL_SEARCH_MOVES):
            improved = False
            for _ in range(self.params.ls_samples):
                if self._spent(prob, start):
                    return cur, cur_m
                cand = LOCAL_SEARCH_MOVES[n](cur, self._rng, prob.ids)
                if cand is None or not prob.feasible(cand):
                    continue
                m = prob.evaluate(cand)
                if m.score < cur_m.score:
                    cur, cur_m, improved = cand, m, True
                    break
            n = 0 if improved else n + 1
        return cur, cur_m
