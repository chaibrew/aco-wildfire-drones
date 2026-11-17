import argparse
import csv
import statistics
import time

from fire_sim.fire import load_fire
from fire_sim.mission import MissionConfig, run_mission
from fire_sim.simulation import smoke_router

from aco_router import ACOParams, ACORouter, FireSimRouter, StrategyName, summary
from baselines import GreedyRouter, VNSParams, VNSRouter


def parse_args():
    parser = argparse.ArgumentParser(description="Compare smoke_router and the four ACO strategies on a FEDS fire.")
    parser.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    parser.add_argument("--fire-id", type=int, default=3885)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--hours", type=float, default=24.0, help="mission length")
    parser.add_argument("--drones", type=int, default=6)
    parser.add_argument("--cell-size", type=float, default=1000.0)
    parser.add_argument("--ants", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--out", required=True, help="results csv")
    return parser.parse_args()


def make_planners(params):
    planners = [
        ("smoke_router", lambda seed: smoke_router),
        ("greedy", lambda seed: GreedyRouter()),
        # VNS gets the same number of plan evaluations as ACO (ants * iterations) per replan
        ("vns", lambda seed: VNSRouter(VNSParams(max_evaluations=params.n_ants * params.max_iterations), seed=seed)),
    ]
    for strategy in StrategyName:
        def make(seed, strategy=strategy):
            return FireSimRouter(ACORouter(params, strategy=strategy, seed=seed))
        planners.append((f"aco_{strategy.value}", make))
    return planners


def main():
    args = parse_args()
    config = MissionConfig(cell_size_m=args.cell_size, drones=args.drones, horizon_s=600)
    planners = make_planners(ACOParams(n_ants=args.ants, max_iterations=args.iterations))

    results = []
    for seed in range(args.seeds):
        field, _ = load_fire(args.gpkg, fire_id=args.fire_id, year=args.year, limit=args.snapshots, seed=seed,
                             cache_dir=args.cache)
        for name, make in planners:
            started = time.time()
            rows, _ = run_mission(field, make(seed), config, stop_s=field.start_s + args.hours * 3600)
            s = summary(rows)
            results.append({"seed": seed, "planner": name, **s, "wall_s": time.time() - started})
            print(f"seed {seed} {name:22s} map J {s['map_J_mean']:.3f}  true J {s['true_J_mean']:.3f}", flush=True)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'planner':22s} {'map J':>8s} {'true J':>8s} {'true overdue':>13s}")
    for name, _ in planners:
        mine = [r for r in results if r["planner"] == name]
        print(f"{name:22s} {statistics.mean(r['map_J_mean'] for r in mine):8.3f} "
              f"{statistics.mean(r['true_J_mean'] for r in mine):8.3f} "
              f"{statistics.mean(r['true_overdue_mean'] for r in mine):13.2f}")
    print(f"\nwrote {args.out}; run scripts/aco_analyse.py on it for paired comparisons")


if __name__ == "__main__":
    main()
