import csv
import glob
import json
import math
import os
import pickle
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import CACHE, FIRE, GPKG, km  # noqa: E402
from fire_sim.fire import load_fire  # noqa: E402
from fire_sim.terrain import fetch_elevation  # noqa: E402

OUT = sys.argv[1]
SOURCES = ("outputs/tuning_missions", "outputs/tuning_missions_d10")
SEEDS = {0}


SETTINGS = {"tuned", "baseline"}


def run_name(setting, strategy, seed, drones):
    return f"{setting}_{strategy}_s{seed}_d{drones}"


def terrain_and_fire():
    field0, sequence = load_fire(GPKG, fire_id=FIRE, year=2018, limit=8, seed=0, cache_dir=CACHE)
    x0, y0, x1, y1 = field0.raster.bounds
    elev = fetch_elevation(field0.raster, field0.crs, os.path.join(CACHE, "dem"))
    gy, gx = np.gradient(elev, field0.raster.res)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    shade = np.sin(np.radians(45)) * np.cos(slope) + np.cos(np.radians(45)) * np.sin(slope) * np.cos(np.radians(315) - aspect)
    shade = np.clip((shade + 0.2) / 1.4, 0, 1)
    rgba = np.ones(shade.shape + (4,))
    rgba[..., :3] = shade[..., None] * 0.55 + 0.4
    plt.imsave(f"{OUT}/hillshade.png", np.flipud(rgba))
    perimeters = []
    for p, s in zip(sequence, field0.snapshot_s):
        if s > 24 * 3600:
            break
        for g in (list(p.geometry.geoms) if p.geometry.geom_type == "MultiPolygon" else [p.geometry]):
            perimeters.append({"h": round(s / 3600, 3), "ring": [km(x, y) for x, y in g.simplify(50).exterior.coords]})
    return km(x0, y0) + km(x1, y1), perimeters, (x0, y0)


def isochrones(seed, origin):
    field, _ = load_fire(GPKG, fire_id=FIRE, year=2018, limit=8, seed=seed, cache_dir=CACHE)
    xs = (np.arange(field.raster.nx) + 0.5) * field.raster.res + origin[0]
    ys = (np.arange(field.raster.ny) + 0.5) * field.raster.res + origin[1]
    cs = plt.contour(xs, ys, field.arrival_s / 3600, levels=np.arange(1, 25, 1))
    out = [{"h": int(lv), "path": [km(x, y) for x, y in seg[::2]]}
           for lv, segs in zip(cs.levels, cs.allsegs) for seg in segs if len(seg) > 4]
    plt.close("all")
    return out


PHEROMONE = "outputs/site_pheromone"


def load_pheromone(rec):
    path = f"{PHEROMONE}/{run_name(rec['setting'], rec['strategy'], rec['seed'], rec['drones'])}.pkl"
    if rec["setting"] != "tuned" or not os.path.exists(path):
        return None
    ph = pickle.load(open(path, "rb"))["stages"]
    same = len(ph) == len(rec["stages"]) and all(a["routes"] == b["routes"] for a, b in zip(ph, rec["stages"]))
    if not same:
        raise SystemExit(f"{path}: re-run routes differ from the mission recording")
    return ph


def export_run(rec):
    table, index = [], {}
    stages, trips = [], defaultdict(list)

    def ref(x, y, risk=-1):
        if (x, y) not in index:
            index[(x, y)] = len(table)
            table.append(km(x, y) + [risk])
        return index[(x, y)]

    ph = load_pheromone(rec)

    def edges(es):
        flat = []
        for x1, y1, x2, y2, r in es:
            flat += [ref(x1, y1), ref(x2, y2), int(round(r * 10))]
        return flat

    for k_stage, st in enumerate(rec["stages"]):
        t = st["t"]
        cells = []
        for x, y, seen, q, risk in st["points"]:
            n = ref(x, y, risk)
            table[n][2] = risk
            stale = 3.0 if seen is None else min((t - seen) / q, 3.0)
            cells += [n, int(round(stale * 20))]
        r = st["row"]
        stages.append({"t": int(t), "c": cells, "true_over": r["true_overdue_count"],
                       "front": r["true_front_points"], "map_over": r["map_overdue_count"],
                       "changed": round(r["affected_union_fraction"], 3), "snap": bool(r["snapshot_delivered"])})
        if ph:
            if ph[k_stage]["ph_change"] is not None:
                stages[-1]["pc"] = edges(ph[k_stage]["ph_change"])
        for k, ((x, y), route) in enumerate(zip(st["drones"], st["routes"])):
            clock = t
            trips[k] += [ref(x, y), int(clock)]
            for px, py in route:
                clock += math.hypot(px - x, py - y) / st["speed"]
                trips[k] += [ref(px, py), int(round(clock))]
                clock += st["service"]
                x, y = px, py
    first = rec["stages"][0]
    return {"points": table, "speed": first["speed"], "service": first["service"], "stages": stages,
            "trips": [trips[k] for k in sorted(trips)]}


def main():
    os.makedirs(f"{OUT}/runs", exist_ok=True)
    os.makedirs(f"{OUT}/fire", exist_ok=True)
    bounds, perimeters, origin = terrain_and_fire()

    summary = []
    for src in SOURCES:
        for r in csv.DictReader(open(f"{src}/missions.csv")):
            if r["setting"] not in SETTINGS:
                continue
            summary.append({"setting": r["setting"], "strategy": r["strategy"], "seed": int(r["seed"]),
                            "drones": int(r["drones"]), "overdue": round(float(r["true_overdue_mean"]), 3),
                            "J": round(float(r["true_J_mean"]), 4), "planning_s": round(float(r["planning_s_mean"]), 3)})

    runs = []
    for old in glob.glob(f"{OUT}/runs/*.json"):
        os.remove(old)
    for src in SOURCES:
        for path in sorted(glob.glob(f"{src}/stages/*.pkl")):
            rec = pickle.load(open(path, "rb"))
            if rec["seed"] not in SEEDS or rec["setting"] not in SETTINGS:
                continue
            name = run_name(rec["setting"], rec["strategy"], rec["seed"], rec["drones"])
            with open(f"{OUT}/runs/{name}.json", "w") as fh:
                json.dump(export_run(rec), fh, separators=(",", ":"))
            runs.append({"id": name, "setting": rec["setting"], "strategy": rec["strategy"], "seed": rec["seed"],
                         "drones": rec["drones"]})
            print("run", name, flush=True)

    for seed in sorted(SEEDS):
        with open(f"{OUT}/fire/s{seed}.json", "w") as fh:
            json.dump(isochrones(seed, origin), fh, separators=(",", ":"))
        print("fire", seed, flush=True)

    with open(f"{OUT}/index.json", "w") as fh:
        json.dump({"bounds": bounds, "perimeters": perimeters, "runs": runs, "summary": summary,
                   "stale_scale": 20}, fh, separators=(",", ":"))
    print("wrote", len(runs), "runs")


if __name__ == "__main__":
    main()
