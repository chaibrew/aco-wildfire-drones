import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment.replay import baseline_planners, run_replay  # noqa: E402
from fire_sim.fire import load_fire  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    p.add_argument("--fire-id", type=int, default=3885)
    p.add_argument("--year", type=int, default=2018)
    p.add_argument("--snapshots", type=int, default=8)
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--drones", type=int, nargs="+", default=[6, 12])
    p.add_argument("--seeds", type=int, default=10, help="number of seeds (0..N-1)")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--cell-size", type=float, default=1000.0)
    p.add_argument("--ants", type=int, default=20)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--vns-evaluations", type=int, default=None, help="default: ants * iterations")
    p.add_argument("--planners", nargs="+", default=["greedy", "vns"], choices=["greedy", "vns"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    evals = a.vns_evaluations or a.ants * a.iterations
    planners = {k: v for k, v in baseline_planners(evals).items() if k in a.planners}

    def field_for_seed(seed):
        return load_fire(a.gpkg, fire_id=a.fire_id, year=a.year, limit=a.snapshots, seed=seed,
                         cache_dir=a.cache)[0]

    paths = run_replay(field_for_seed, planners, a.drones, list(range(a.seeds)), a.hours, a.out,
                       cell_size_m=a.cell_size, resume=a.resume)
    print(f"\nwrote {paths['stages']} and {paths['summary']}")


if __name__ == "__main__":
    main()