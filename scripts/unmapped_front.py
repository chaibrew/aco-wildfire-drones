import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from baselines.greedy import greedy_routes  # noqa: E402
from fire_sim.fire import load_fire  # noqa: E402
from fire_sim.mission import FrontMonitor, MissionConfig, run_mission  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="List the true-front cells that are not on the map, with the distance to the nearest map point.")
    p.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    p.add_argument("--fire-id", type=int, default=3885)
    p.add_argument("--year", type=int, default=2018)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--drones", type=int, default=10)
    p.add_argument("--hours", type=float, nargs="+", default=[2, 6, 10])
    a = p.parse_args()
    field, _ = load_fire(a.gpkg, fire_id=a.fire_id, year=a.year, limit=8, seed=a.seed, cache_dir="data/cache")
    cfg = MissionConfig(cell_size_m=1000, drones=a.drones, horizon_s=600)
    monitor = FrontMonitor(field, cell_size_m=cfg.cell_size_m, base_revisit_s=cfg.base_revisit_s,
                           rate_tiers_mpm=cfg.rate_tiers_mpm)
    wanted = {round(h * 3600 / cfg.horizon_s) for h in a.hours}

    def record(stage, fire_map, points, routes):
        if stage not in wanted:
            return
        t = field.start_s + (stage + 1) * cfg.horizon_s
        true_points = monitor.points(t)
        missing = [i for i in true_points if i not in points]
        xy = np.array([(q.x, q.y) for q in points.values()])
        far = sorted((float(np.hypot(xy[:, 0] - true_points[i].x, xy[:, 1] - true_points[i].y).min()),
                      true_points[i].risk) for i in missing)
        print(f"hour {stage * cfg.horizon_s / 3600:4.1f}: true front {len(true_points)}, map {len(points)}, "
              f"true front not on the map {len(missing)}")
        print("    distance to the nearest map point (m), risk: "
              + ", ".join(f"{d:.0f} (r{r})" for d, r in far))

    run_mission(field, greedy_routes, cfg, stop_s=field.start_s + (max(a.hours) + 0.5) * 3600, record=record)


if __name__ == "__main__":
    main()
