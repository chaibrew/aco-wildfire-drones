"""Rerun full Camp Fire missions for Greedy and VNS at several drone counts, saving every stage.

    python scripts/mission_replay.py --drones 6 12 --seeds 10 --hours 24 --out outputs/replay-24h

Then look at a stage:   python scripts/inspect_stage.py outputs/replay-24h/stages.csv --stage 40
Scan for jumps:         python scripts/inspect_stage.py outputs/replay-24h/stages.csv --scan
Add --resume to continue an interrupted run.  VNS gets --vns-evaluations plan evaluations per
replan (default ants * iterations, the same budget unit as ACO).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment.replay import baseline_planners, run_replay  # noqa: E402
from fire_sim.fire import load_fire  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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