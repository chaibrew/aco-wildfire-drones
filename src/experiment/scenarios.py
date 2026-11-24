from __future__ import annotations

import math
import random

from fire_sim.models import MonitoringPoint
from fire_sim.simulation import Drone, PlanningState, coverage

CELL_M = 1000.0
BASE_REVISIT_S = 900.0


def synthetic_state(n_points: int = 60, n_drones: int = 6, seed: int = 0, *, cell_size_m: float = CELL_M,
                    sensor_radius_m: float | None = None, horizon_s: float = 600.0, speed_mps: float = 15.0,
                    service_s: float = 5.0, risk_weights=(0.5, 0.3, 0.2)) -> PlanningState:
    rng = random.Random(seed)
    radius = cell_size_m if sensor_radius_m is None else sensor_radius_m
    r0 = n_points * cell_size_m / (2 * math.pi)
    lobes = [(rng.randint(2, 5), rng.uniform(0.05, 0.2), rng.uniform(0, 2 * math.pi)) for _ in range(2)]
    pts: dict = {}
    k = 0
    while len(pts) < n_points:
        a = 2 * math.pi * k / (n_points * 1.2)
        k += 1
        r = r0 * (1 + sum(amp * math.sin(m * a + ph) for m, amp, ph in lobes))
        cid = (math.floor(r * math.cos(a) / cell_size_m), math.floor(r * math.sin(a) / cell_size_m))
        if cid in pts:
            continue
        risk = rng.choices((1, 2, 3), weights=risk_weights)[0]
        pts[cid] = MonitoringPoint(cid, (cid[0] + 0.5) * cell_size_m, (cid[1] + 0.5) * cell_size_m, risk,
                                   BASE_REVISIT_S / risk)
    ids = sorted(pts, key=lambda c: math.atan2(pts[c].y, pts[c].x))
    drones = tuple(Drone(d, pts[ids[(d * len(ids)) // n_drones]].x, pts[ids[(d * len(ids)) // n_drones]].y)
                   for d in range(n_drones))
    last_seen = {i: -rng.uniform(0.0, pts[i].revisit_s) for i in pts}
    return PlanningState(0.0, horizon_s, speed_mps, service_s, drones, pts, last_seen,
                         coverage(pts, radius) if radius > 0 else None)


def captured_states(gpkg: str, fire_id: int, year: int, *, snapshots: int = 8, cache: str = "data/cache",
                    seed: int = 0, cell_size_m: float = CELL_M, drones: int = 6, horizon_s: float = 600.0,
                    stages: tuple[int, ...] = (6, 40, 80)) -> dict[int, PlanningState]:
    from fire_sim.fire import load_fire
    from fire_sim.mission import MissionConfig, run_mission

    from baselines.greedy import greedy_routes

    field, _ = load_fire(gpkg, fire_id=fire_id, year=year, limit=snapshots, seed=seed, cache_dir=cache)
    cfg = MissionConfig(cell_size_m=cell_size_m, drones=drones, horizon_s=horizon_s)
    keep: dict[int, PlanningState] = {}
    counter = [0]

    def recorder(state: PlanningState):
        if counter[0] in stages:
            keep[counter[0]] = state
        counter[0] += 1
        return greedy_routes(state)

    run_mission(field, recorder, cfg, stop_s=field.start_s + (max(stages) + 1) * horizon_s)
    return keep
