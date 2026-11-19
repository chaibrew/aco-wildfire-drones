"""Re-run the seed-0 tuned ACO missions and record the pheromone at every replan, for the site's replay map.

The 24 h mission recordings (scripts/run_tuned_missions.py) keep points, drones and routes but not pheromone.
This script repeats the same missions (same fire, seed, parameters and budget, so the same plans) and stores,
per replan, the strongest pheromone edges at two moments:

  ph_change  right after the change is applied (carry-over by point id, then the strategy's reset)
  ph_plan    after the replan's iterations, i.e. the pheromone the plan was built from

Pheromone is shown relative to the MMAS floor, not tau0: with every ant depositing and tau clipped to
[0.024 tau_max, tau_max], tau_max sits below tau0, so learned edges are a multiple of the floor while unused
edges sit on it. Edges exactly at tau0 were just reset (or are new) and carry nothing learned, so they are
skipped. Each snapshot is the top-k outgoing point-to-point edges per point with tau/floor >= FLOOR_X, as
(x1, y1, x2, y2, tau/floor) in metres. Routes are stored too so export_missions.py can check the re-run matches the recording.

    python scripts/site/record_pheromone.py outputs/site_pheromone
"""
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from fire_sim.fire import load_fire  # noqa: E402
from fire_sim.mission import MissionConfig, run_mission  # noqa: E402

from aco_router import ACOParams, ACORouter, FireSimRouter, StrategyName  # noqa: E402

ANTS, ITERATIONS = 20, 60  # as scripts/run_tuned_missions.py
GPKG, CACHE = "data/raw/LargeFires_2018.gpkg", "data/cache"
SEED = 0
DRONES = (6, 10, 12)
TOP_K, FLOOR_X = 2, 3.0


def top_edges(ph, tau0, coords):
    idx = [i for key, i in ph.index.items() if key in coords]  # drop drone start nodes
    if not idx:
        return []
    sub = ph.tau[np.ix_(idx, idx)]
    learned = ~np.isclose(sub, tau0, rtol=1e-9)
    if not learned.any():
        return []  # all reset (full reset at a change)
    floor = sub[learned].min()
    out = []
    for a, i in enumerate(idx):
        row = np.where(learned[a], sub[a] / floor, 0.0)
        row[a] = 0.0
        for b in np.argsort(row)[::-1][:TOP_K]:
            if row[b] >= FLOOR_X:
                (x1, y1), (x2, y2) = coords[ph.keys[i]], coords[ph.keys[idx[b]]]
                out.append((x1, y1, x2, y2, round(float(row[b]), 1)))
    return out


class Recorder:
    def __init__(self, router: FireSimRouter):
        self.router, self.stages = router, []
        self._at_change = None
        aco = router.router
        on_change = aco.strategy.on_change

        def wrapped(ph, change, tau0):
            on_change(ph, change, tau0)
            self._at_change = (ph.copy(), tau0)

        aco.strategy.on_change = wrapped

    def __call__(self, state):
        self._at_change = None
        routes = self.router(state)
        coords = {i: (p.x, p.y) for i, p in state.points.items()}
        aco = self.router.router
        ph_change = top_edges(self._at_change[0], self._at_change[1], coords) if self._at_change else None
        self.stages.append({
            "t": state.time_s,
            "routes": [[(state.points[i].x, state.points[i].y) for i in routes[d.id]] for d in state.drones],
            "ph_change": ph_change,
            "ph_plan": top_edges(aco.pheromone, aco.tau0, coords),
        })
        return routes


def job(args):
    out, strategy, drones = args
    field, _ = load_fire(GPKG, fire_id=3885, year=2018, limit=8, seed=SEED, cache_dir=CACHE)
    params = ACOParams().with_(n_ants=ANTS, max_iterations=ITERATIONS)
    rec = Recorder(FireSimRouter(ACORouter(params, strategy=StrategyName(strategy), seed=SEED)))
    t0 = time.time()
    run_mission(field, rec, MissionConfig(cell_size_m=1000.0, drones=drones, horizon_s=600),
                stop_s=field.start_s + 24 * 3600)
    name = f"tuned_{strategy}_s{SEED}_d{drones}"
    with open(f"{out}/{name}.pkl", "wb") as fh:
        pickle.dump({"strategy": strategy, "drones": drones, "seed": SEED, "stages": rec.stages}, fh)
    return name, time.time() - t0


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    jobs = [(out, s.value, d) for d in DRONES for s in StrategyName]
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 9))) as pool:
        for i, (name, wall) in enumerate(pool.imap_unordered(job, jobs), 1):
            print(f"[{i}/{len(jobs)} {time.time() - t0:5.0f}s] {name} ({wall:.0f}s)", flush=True)
    print("done", f"{time.time() - t0:.0f}s")
