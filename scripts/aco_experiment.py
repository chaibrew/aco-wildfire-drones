import argparse
import csv
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from fire_sim.fire import load_fire
from fire_sim.mission import MissionConfig, run_mission
from fire_sim.simulation import smoke_router

from aco_router import ACOParams, ACORouter, FireSimRouter, StrategyName, summary
from baselines import GreedyRouter, VNSParams, VNSRouter

PLANNERS = ("smoke_router", "greedy", "vns") + tuple(f"aco_{s.value}" for s in StrategyName)
REPORT = [("map_J_mean", "map J", True), ("true_J_mean", "true J", True),
          ("true_overdue_mean", "true overdue", True), ("perimeter_error_mean_m", "perim err m", True),
          ("growth_found_mean", "growth found", False)]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare smoke_router, greedy, VNS and the four ACO strategies on "
                                                 "a FEDS fire.")
    parser.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    parser.add_argument("--fire-id", type=int, default=3885)
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--hours", type=float, default=24.0, help="mission length")
    parser.add_argument("--drones", type=int, default=6)
    parser.add_argument("--cell-size", type=float, default=1000.0)
    parser.add_argument("--ants", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--params", default=None,
                        help="JSON of ACO parameters (e.g. outputs/tuning/defaults_params.json); "
                             "--ants and --iterations still set the budget")
    parser.add_argument("--trigger", choices=("any", "widespread"), default="any",
                        help="when a pheromone strategy acts: any change (original) or only widespread changes "
                             "(> 50%% of points, e.g. a new FEDS snapshot)")
    parser.add_argument("--selective-radius", type=float, default=0.0,
                        help="selective reset also resets points within this many metres of a change")
    parser.add_argument("--vns-evals", type=int, default=None,
                        help="VNS plan evaluations per replan (default: ants * iterations, the ACO budget)")
    parser.add_argument("--planners", nargs="+", choices=PLANNERS, default=list(PLANNERS))
    parser.add_argument("--workers", type=int, default=1, help="(seed, planner) missions run in parallel")
    parser.add_argument("--stages-out", default=None,
                        help="also write every stage's row (map accuracy, true J, ...) to this csv")
    parser.add_argument("--out", required=True, help="results csv")
    return parser.parse_args()


def make_router(name, params, seed, vns_evals=None):
    if name == "smoke_router":
        return smoke_router
    if name == "greedy":
        return GreedyRouter()
    if name == "vns":
        return VNSRouter(VNSParams(max_evaluations=vns_evals or params.n_ants * params.max_iterations), seed=seed)
    return FireSimRouter(ACORouter(params, strategy=StrategyName(name.removeprefix("aco_")), seed=seed))


def run_one(job):
    args, params, seed, name = job
    field, _ = load_fire(args.gpkg, fire_id=args.fire_id, year=args.year, limit=args.snapshots, seed=seed,
                         cache_dir=args.cache)
    config = MissionConfig(cell_size_m=args.cell_size, drones=args.drones, horizon_s=600)
    started = time.time()
    rows, _ = run_mission(field, make_router(name, params, seed, args.vns_evals), config,
                          stop_s=field.start_s + args.hours * 3600)
    result = {"seed": seed, "planner": name, **summary(rows), "wall_s": time.time() - started}
    stages = [{"seed": seed, "planner": name, **r} for r in rows]
    return result, stages


def main():
    args = parse_args()
    base = ACOParams()
    if args.params:
        from tuning.space import load_params
        base = load_params(args.params)
    params = base.with_(n_ants=args.ants, max_iterations=args.iterations, change_trigger=args.trigger,
                        selective_radius_m=args.selective_radius)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    jobs = [(args, params, seed, name) for seed in seeds for name in args.planners]
    results, stages = [], []

    def collect(out):
        result, rows = out
        results.append(result)
        stages.extend(rows)
        print(f"seed {result['seed']} {result['planner']:22s} map J {result['map_J_mean']:.3f}  "
              f"true J {result['true_J_mean']:.3f}  perimeter error {result['perimeter_error_mean_m']:.0f} m",
              flush=True)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            for out in pool.map(run_one, jobs):
                collect(out)
    else:
        for job in jobs:
            collect(run_one(job))

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    if args.stages_out:
        with open(args.stages_out, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(stages[0]))
            writer.writeheader()
            writer.writerows(stages)

    print(f"\n{'planner':22s} " + " ".join(f"{label:>13s}" for _, label, _ in REPORT))
    for name in args.planners:
        mine = [r for r in results if r["planner"] == name]
        print(f"{name:22s} " + " ".join(f"{statistics.mean(r[key] for r in mine):13.3f}" for key, _, _ in REPORT))
    print(f"\nwrote {args.out}; run scripts/aco_analyse.py on it for paired comparisons")


if __name__ == "__main__":
    main()
