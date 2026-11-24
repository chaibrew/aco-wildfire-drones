from __future__ import annotations

import heapq
import math

from fire_sim.simulation import PlanningState

Routes = dict[int, list]

_EPS = 1e-9


def greedy_routes(state: PlanningState) -> Routes:
    routes: Routes = {d.id: [] for d in state.drones}
    end = state.time_s + state.horizon_s
    coverage = state.coverage or {}
    reserved = dict(state.last_seen)
    queue = [(state.time_s, d.id, d.x, d.y) for d in state.drones]
    heapq.heapify(queue)
    ids = sorted(state.points)

    while queue:
        now, drone_id, x, y = heapq.heappop(queue)
        best = None
        for i in ids:
            p = state.points[i]
            travel = math.hypot(p.x - x, p.y - y) / state.speed_mps + state.service_s
            arrival = now + travel
            if arrival > end + _EPS:
                continue
            ratio = max(0.0, arrival - reserved[i]) / p.revisit_s
            key = (ratio / travel, -travel)
            if best is None or key > best[0]:
                best = (key, i, arrival)
        if best is None:
            continue
        _, i, arrival = best
        p = state.points[i]
        routes[drone_id].append(i)
        for j in coverage.get(i, (i,)):
            if j in reserved:
                reserved[j] = max(reserved[j], arrival)
        heapq.heappush(queue, (arrival, drone_id, p.x, p.y))
    return routes


class GreedyRouter:
    name = "greedy"

    def __call__(self, state: PlanningState) -> Routes:
        return greedy_routes(state)
