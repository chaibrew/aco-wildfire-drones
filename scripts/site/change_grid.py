import argparse
import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402

from aco_router import ACOParams, ACORouter, diff_instances, state_to_instance  # noqa: E402
from aco_router.pheromone import node_keys  # noqa: E402
from aco_router.strategies import make_strategy  # noqa: E402
from baselines import GreedyRouter, VNSRouter  # noqa: E402
from experiment import captured_states  # noqa: E402
from experiment.perturb import ChangeSpec, apply_change  # noqa: E402
from experiment.runner import STRATEGY_ORDER, ExperimentConfig, score_routes, warm_router  # noqa: E402

from common import GPKG, CACHE, FIRE, km  # noqa: E402

SETTINGS = {"tuned": ACOParams()}
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60)
LOCALITIES = ("concentrated", "distributed")
DETAIL_TRIALS = 2
STATES: dict = {}


def top_edges(ph, tau0, index, k=2, floor=1.5):
    out = []
    for key, i in ph.index.items():
        if key not in index:
            continue
        row = ph.tau[i] / tau0
        for j in np.argsort(row)[::-1][:k]:
            other = ph.keys[j]
            if other in index and row[j] >= floor:
                out.append([index[key], index[other], round(float(row[j]), 1)])
    return out


def points_view(state, affected=frozenset()):
    ids = sorted(state.points)
    index = {pid: n for n, pid in enumerate(ids)}
    pts = []
    for pid in ids:
        p = state.points[pid]
        seen = state.last_seen.get(pid)
        stale = 3.0 if seen is None else (state.time_s - seen) / p.revisit_s
        pts.append(km(p.x, p.y) + [p.risk, round(min(stale, 3.0), 2), int(pid in affected)])
    return ids, index, pts


def job(args):
    stage, setting, seed, fraction, locality = args
    base = STATES[stage]
    cfg = ExperimentConfig(params=SETTINGS[setting], ants=20, iterations=60)
    warmed = warm_router(base, cfg, seed)
    snap = warmed.snapshot()
    changed, info = apply_change(base, ChangeSpec(fraction, locality, "mixed"), seed + 10_000,
                                 cell_size_m=cfg.cell_size_m)
    inst0, inst1 = state_to_instance(base), state_to_instance(changed)
    change = diff_instances(inst0, inst1)
    replan_seed = seed + 20_000
    detail = seed < DETAIL_TRIALS

    ids1, index1, pts1 = points_view(changed, change.affected)
    removed = [km(base.points[p].x, base.points[p].y) for p in sorted(change.removed) if p in base.points]
    drone_ids = [d.id for d in changed.drones]
    res = {"stage": stage, "setting": setting, "trial": seed, "fraction": fraction, "locality": locality,
           "realised": round(info.realised_fraction, 3), "dispersion": round(info.dispersion, 3),
           "added": info.added, "removed": info.removed, "changed": info.changed, "planners": {}}

    def add(name, routes, runtime, ph=None, ph_after=None, tau0=None):
        s = score_routes(changed, routes)
        row = {"margin": round(s["margin"], 4), "total_overdue": round(s["total_overdue"], 4),
               "overdue_count": s["overdue_count"], "objective": round(s["objective"], 4),
               "distance_km": round(s["total_distance_m"] / 1000, 2), "runtime_s": round(runtime, 3)}
        if detail:
            row["routes"] = [[index1[p] for p in routes.get(d, [])] for d in drone_ids]
            if ph is not None:
                row["ph_change"] = top_edges(ph, tau0, index1)
                row["ph_after"] = top_edges(ph_after, tau0, index1)
        res["planners"][name] = row

    t0 = time.perf_counter()
    add("greedy", GreedyRouter()(changed), time.perf_counter() - t0)
    v = VNSRouter(cfg.vns_params(), seed=replan_seed).plan(changed)
    add("vns", v.routes, v.runtime_s)
    for strategy in STRATEGY_ORDER:
        router = ACORouter(cfg.aco_params(), strategy=strategy, seed=replan_seed)
        router.restore(snap)
        router.reseed(replan_seed)
        ph = snap.pheromone.remapped(node_keys(inst1), snap.tau0)
        make_strategy(strategy).on_change(ph, change, snap.tau0)
        result = router.plan(inst1, change)
        add(f"aco_{strategy.value}", result.plan.for_drones(inst1.drone_ids), result.runtime_s,
            ph, router.pheromone, snap.tau0)

    if detail:
        _, index0, pts0 = points_view(base)
        routes0 = warmed.plan(inst0).plan.for_drones(inst0.drone_ids)
        res["before"] = {"points": pts0, "pheromone": top_edges(snap.pheromone, snap.tau0, index0),
                         "routes": [[index0[p] for p in routes0.get(d, [])] for d in drone_ids]}
        res["after"] = {"points": pts1, "removed": removed}
        res["drones"] = [km(d.x, d.y) for d in changed.drones]
        res["time_h"] = round(changed.time_s / 3600, 2)
    return res


def init(stages, drones):
    STATES.update(captured_states(GPKG, FIRE, 2018, snapshots=8, cache=CACHE, drones=drones, stages=tuple(stages)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--stages", type=int, nargs="+", default=[40, 80])
    p.add_argument("--drones", type=int, default=6)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = list(product(a.stages, SETTINGS, range(a.trials), FRACTIONS, LOCALITIES))
    print(len(jobs), "paired trials", flush=True)
    results, t0 = [], time.time()
    with Pool(a.workers, initializer=init, initargs=(a.stages, a.drones)) as pool:
        for n, r in enumerate(pool.imap_unordered(job, jobs), 1):
            results.append(r)
            if n % 20 == 0 or n == len(jobs):
                print(f"{n}/{len(jobs)} done, {time.time() - t0:.0f} s", flush=True)
    results.sort(key=lambda r: (r["stage"], r["setting"], r["trial"], r["fraction"], r["locality"]))
    with open(f"{a.out}/change_grid.json", "w") as fh:
        json.dump({"drones": a.drones, "fractions": FRACTIONS, "localities": LOCALITIES, "trials": results}, fh,
                  separators=(",", ":"))
    print("wrote", f"{a.out}/change_grid.json")


if __name__ == "__main__":
    main()
