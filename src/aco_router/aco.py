from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .changes import diff_instances, with_neighbourhood
from .config import (
    ChangeScope,
    COST_EPS,
    DEFAULT_ACO_PARAMS,
    DISTANCE_EPS_M,
    HORIZON_TOLERANCE_S,
    ACOParams,
    BudgetKind,
    StrategyName,
    Tau0Mode,
)
from .evaluator import evaluate, validate_plan
from .model import ChangeEvent, Metrics, MissionInstance, Plan, PlanResult, TraceEntry
from .pheromone import PheromoneMatrix, node_keys
from .strategies import PheromoneStrategy, make_strategy

Routes = list[list[int]]
Evaluator = Callable[[MissionInstance, Plan], Metrics]


def _shared_evaluator(instance: MissionInstance, plan: Plan) -> Metrics:
    return evaluate(instance, plan, validate=False)


class _Context:
    def __init__(self, instance: MissionInstance, params: ACOParams):
        self.instance = instance
        self.keys = node_keys(instance)
        self.n_drones = nd = len(instance.drones)
        self.n_points = len(instance.points)
        self.n_nodes = nd + self.n_points
        self.drone_ids = instance.drone_ids
        self.point_ids = instance.point_ids
        self.t0 = instance.time
        self.end = instance.end_time
        self.service = instance.service_s
        xs = np.array([d.x for d in instance.drones] + [p.x for p in instance.points], dtype=float)
        ys = np.array([d.y for d in instance.drones] + [p.y for p in instance.points], dtype=float)
        self.dist = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        self.speed = np.array([d.speed for d in instance.drones], dtype=float)
        self.q = np.concatenate([np.full(nd, np.inf), [p.q for p in instance.points]])
        self.last_seen = np.concatenate([np.full(nd, -np.inf), [instance.last_seen(p) for p in instance.points]])
        self.weight = (1.0 / np.maximum(self.dist, DISTANCE_EPS_M)) ** params.beta
        index = {pid: nd + k for k, pid in enumerate(self.point_ids)}
        self.cover = [
            np.array([index[c] for c in instance.observed_by(pid) if c in index], dtype=int) for pid in self.point_ids
        ]

    def to_plan(self, routes: Routes) -> Plan:
        nd = self.n_drones
        return Plan({self.drone_ids[d]: tuple(self.point_ids[j - nd] for j in r) for d, r in enumerate(routes)})

    def edge_arrays(self, routes: Routes) -> tuple[np.ndarray, np.ndarray]:
        rows: list[int] = []
        cols: list[int] = []
        for d, r in enumerate(routes):
            if not r:
                continue
            rows.append(d)
            cols.append(r[0])
            for a, b in zip(r, r[1:]):
                rows += (a, b)
                cols += (b, a)
        return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


@dataclass(frozen=True)
class RouterState:
    pheromone: PheromoneMatrix | None
    tau0: float | None
    last_instance: MissionInstance | None
    iterations_since_change: int | None
    rng_state: dict


class ACORouter:
    def __init__(
        self,
        params: ACOParams = DEFAULT_ACO_PARAMS,
        strategy: StrategyName | str | PheromoneStrategy = StrategyName.SELECTIVE_RESET,
        seed: int = 0,
        name: str | None = None,
        evaluator: Evaluator | None = None,
    ):
        self.params = params
        self.strategy = make_strategy(strategy)
        self.evaluator: Evaluator = evaluator or _shared_evaluator
        self.name = name or f"aco_{self.strategy.name.value}"
        self._rng = np.random.default_rng(seed)
        self._pheromone: PheromoneMatrix | None = None
        self._tau0: float | None = None
        self._last_instance: MissionInstance | None = None
        self._iters_since_change: int | None = None
        self.last_change: ChangeEvent | None = None


    @property
    def pheromone(self) -> PheromoneMatrix | None:
        return self._pheromone

    @property
    def tau0(self) -> float | None:
        return self._tau0

    def reseed(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def snapshot(self) -> RouterState:
        return RouterState(
            pheromone=None if self._pheromone is None else self._pheromone.copy(),
            tau0=self._tau0,
            last_instance=self._last_instance,
            iterations_since_change=self._iters_since_change,
            rng_state=copy.deepcopy(self._rng.bit_generator.state),
        )

    def restore(self, state: RouterState) -> None:
        self._pheromone = None if state.pheromone is None else state.pheromone.copy()
        self._tau0 = state.tau0
        self._last_instance = state.last_instance
        self._iters_since_change = state.iterations_since_change
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = copy.deepcopy(state.rng_state)


    def plan(self, instance: MissionInstance, change: ChangeEvent | None = None) -> PlanResult:
        start = time.perf_counter()
        ctx = _Context(instance, self.params)
        self._prepare_pheromone(ctx, change)

        if ctx.n_points == 0:
            plan = Plan.empty(instance)
            return PlanResult(plan, self.evaluator(instance, plan), self.name, time.perf_counter() - start, 0, ())

        best_m: Metrics | None = None
        best_routes: Routes | None = None
        trace: list[TraceEntry] = []
        iteration = 0
        while True:
            evaluated = []
            for _ in range(self.params.n_ants):
                routes = self._construct(ctx)
                m = self.evaluator(instance, ctx.to_plan(routes))
                evaluated.append((m, routes))
                if best_m is None or m.score < best_m.score:
                    best_m, best_routes = m, routes
            self._update(ctx, evaluated, best_m.cost)
            iteration += 1
            if self._iters_since_change is not None:
                self._iters_since_change += 1
            trace.append(
                TraceEntry(iteration, time.perf_counter() - start, best_m.objective, best_m.margin, best_m.total_overdue)
            )
            if self._budget_spent(iteration, start):
                break

        plan = ctx.to_plan(best_routes)
        validate_plan(instance, plan)
        metrics = self.evaluator(instance, plan)
        return PlanResult(plan, metrics, self.name, time.perf_counter() - start, iteration, tuple(trace))

    def _prepare_pheromone(self, ctx: _Context, change: ChangeEvent | None) -> None:
        if self._pheromone is None:
            self._tau0 = self._initial_tau0(ctx)
            self._pheromone = PheromoneMatrix(ctx.keys, self._tau0)
            self._iters_since_change = None
            self.last_change = None
        else:
            if change is None:
                change = diff_instances(self._last_instance, ctx.instance)
            self._pheromone = self._pheromone.remapped(ctx.keys, self._tau0)
            if change.is_empty or not self._triggers(change):
                self._iters_since_change = None
            else:
                if self.params.selective_radius_m > 0:
                    change = with_neighbourhood(change, self._last_instance, ctx.instance,
                                                self.params.selective_radius_m)
                self.strategy.on_change(self._pheromone, change, self._tau0)
                self._iters_since_change = 0
            self.last_change = change
        self._last_instance = ctx.instance

    def _triggers(self, change: ChangeEvent) -> bool:
        return self.params.change_trigger == "any" or change.scope is ChangeScope.WIDESPREAD

    def _initial_tau0(self, ctx: _Context) -> float:
        p = self.params
        if p.tau0_mode is Tau0Mode.CONSTANT or ctx.n_points == 0:
            return p.tau0_constant
        greedy = self._construct(ctx, greedy=True)
        m = self.evaluator(ctx.instance, ctx.to_plan(greedy))
        return p.n_ants * p.deposit_q / max(m.cost, COST_EPS)

    def _budget_spent(self, iteration: int, start: float) -> bool:
        p = self.params
        if p.budget_kind is BudgetKind.ITERATIONS:
            return iteration >= p.max_iterations
        return time.perf_counter() - start >= p.time_budget_s

    def _construct(self, ctx: _Context, greedy: bool = False) -> Routes:
        p = self.params
        nd = ctx.n_drones
        routes: Routes = [[] for _ in range(nd)]
        if ctx.n_points == 0:
            return routes
        clock = np.full(nd, ctx.t0)
        cur = np.arange(nd)
        active = np.ones(nd, dtype=bool)
        # last planned observation of each point, so ants avoid revisiting what they just saw
        reserved = ctx.last_seen.copy()
        tau = None if greedy else self._pheromone.tau
        limit = ctx.end + HORIZON_TOLERANCE_S

        while active.any():
            # the drone that is free first chooses next
            d = int(np.argmin(np.where(active, clock, np.inf)))
            i = int(cur[d])
            arrive = clock[d] + (ctx.dist[i, nd:] / ctx.speed[d] + ctx.service)
            feasible = (arrive <= limit) & (arrive > clock[d])
            if i >= nd:
                feasible[i - nd] = False  # no immediate repeat of the same point
            cand = np.flatnonzero(feasible)
            if cand.size == 0:
                active[d] = False
                continue
            nodes = cand + nd
            # how overdue each candidate would be on arrival, relative to its revisit period
            urgency = np.maximum(0.0, arrive[cand] - reserved[nodes]) / ctx.q[nodes]
            w = ctx.weight[i, nodes] * (p.urgency_floor + urgency) ** p.urgency_weight
            if greedy:
                pick = int(np.argmax(w))
            else:
                t = tau[i, nodes]
                w = w * (t if p.alpha == 1.0 else t**p.alpha)
                cum = np.cumsum(w)
                total = cum[-1]
                if total > 0 and np.isfinite(total):
                    pick = min(int(np.searchsorted(cum, self._rng.random() * total, side="right")), cand.size - 1)
                else:
                    pick = int(self._rng.integers(cand.size))
            j = int(nodes[pick])
            routes[d].append(j)
            clock[d] = arrive[cand[pick]]
            cur[d] = j
            seen = ctx.cover[j - nd]  # one visit also observes nearby points
            reserved[seen] = np.maximum(reserved[seen], clock[d])
        return routes

    def _update(self, ctx: _Context, evaluated: list[tuple[Metrics, Routes]], best_cost: float) -> None:
        p = self.params
        pher = self._pheromone
        pher.evaporate(self.strategy.evaporation_rate(p, self._iters_since_change))
        for m, routes in evaluated:
            r, c = ctx.edge_arrays(routes)
            if r.size:
                pher.deposit(r, c, p.deposit_q / max(m.cost, COST_EPS))
        if p.use_bounds:
            # mmas floor: without it unused edges decay to ~0 and urgent points become unreachable
            tau_max = p.n_ants * p.deposit_q / (max(p.rho, 1e-12) * max(best_cost, COST_EPS))
            pher.clip(tau_max * p.tau_min_ratio, tau_max)
