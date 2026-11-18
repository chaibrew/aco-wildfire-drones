import csv
import os
import pickle
import statistics
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fire_sim.fire import load_fire  # noqa: E402
from fire_sim.mission import MissionConfig, run_mission  # noqa: E402

from aco_router import ACOParams, ACORouter, FireSimRouter, StrategyName, summary  # noqa: E402
from tuning import UNTUNED_PARAMS  # noqa: E402

OUT = sys.argv[1]
SEEDS = range(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
DRONES = (6, 12)
SETTINGS = {"untuned": UNTUNED_PARAMS, "tuned": ACOParams()}
GPKG, CACHE = "data/raw/LargeFires_2018.gpkg", "data/cache"


class Recorder:
    def __init__(self, inner):
        self.inner, self.stages = inner, []

    def __call__(self, state):
        routes = self.inner(state)
        self.stages.append({
            "t": state.time_s,
            "points": [(p.x, p.y, state.last_seen[i], p.revisit_s, p.risk) for i, p in state.points.items()],
            "drones": [(d.x, d.y) for d in state.drones],
            "routes": [[(state.points[i].x, state.points[i].y) for i in routes[d.id]] for d in state.drones],
            "speed": state.speed_mps, "service": state.service_s})
        return routes


def job(args):
    setting, strategy, seed, drones = args
    path = f"{OUT}/stages/{setting}_{strategy}_s{seed}_d{drones}.pkl"
    params = SETTINGS[setting].with_(n_ants=20, max_iterations=60)
    field, _ = load_fire(GPKG, fire_id=3885, year=2018, limit=8, seed=seed, cache_dir=CACHE)
    rec = Recorder(FireSimRouter(ACORouter(params, strategy=StrategyName(strategy), seed=seed)))
    started = time.time()
    rows, _ = run_mission(field, rec, MissionConfig(cell_size_m=1000.0, drones=drones, horizon_s=600),
                          stop_s=field.start_s + 24 * 3600)
    wall = time.time() - started
    for st, row in zip(rec.stages, rows):
        st["row"] = row
    with open(path, "wb") as fh:
        pickle.dump({"setting": setting, "strategy": strategy, "seed": seed, "drones": drones,
                     "params": params, "stages": rec.stages}, fh)
    s = summary(rows)
    return {"setting": setting, "strategy": strategy, "seed": seed, "drones": drones, **s,
            "true_max_gap_mean": statistics.mean(r["true_max_gap_s"] for r in rows),
            "true_overdue_total": sum(r["true_overdue_count"] for r in rows), "wall_s": wall}


if __name__ == "__main__":
    os.makedirs(f"{OUT}/stages", exist_ok=True)
    jobs = [(st, sg.value, seed, d) for d in DRONES for seed in SEEDS for sg in StrategyName for st in SETTINGS]
    results = []
    t0 = time.time()
    with Pool(int(os.environ.get("WORKERS", 9))) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            results.append(r)
            print(f"[{i}/{len(jobs)} {time.time() - t0:5.0f}s] {r['setting']:7s} {r['strategy']:16s} seed {r['seed']} "
                  f"d{r['drones']:2d} true overdue {r['true_overdue_mean']:.2f}", flush=True)
            with open(f"{OUT}/missions.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0]))
                w.writeheader()
                w.writerows(sorted(results, key=lambda r: (r["drones"], r["strategy"], r["seed"], r["setting"])))
    print("done", f"{time.time() - t0:.0f}s")
