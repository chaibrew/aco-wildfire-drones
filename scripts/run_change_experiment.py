import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment import ExperimentConfig, captured_states, run_from_states, run_synthetic, write_csv  # noqa: E402
from experiment.runner import PLANNERS, RESULT_FIELDS, TRACE_FIELDS  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=("synthetic", "feds"), default="synthetic")
    p.add_argument("--phase", choices=("pilot", "final"), default="final")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.40, 0.60])
    p.add_argument("--localities", nargs="+", default=["concentrated", "distributed"],
                   choices=["concentrated", "distributed"])
    p.add_argument("--kind", choices=("risk", "advance", "mixed"), default="mixed")
    p.add_argument("--direction", choices=("cycle", "up"), default="cycle",
                   help="risk changes: cycle 1->2->3->1 (original) or up one tier (risk 3 advances instead)")
    p.add_argument("--selective-radius", type=float, default=None,
                   help="selective reset also resets points within this many metres of a change (default: params)")
    p.add_argument("--planners", nargs="+", default=list(PLANNERS), choices=list(PLANNERS))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5, help="replans on the base state before the change")
    p.add_argument("--ants", type=int, default=20)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--vns-evals", type=int, default=None, help="default: ants * iterations (equal budget)")
    p.add_argument("--points", type=int, default=60, help="synthetic: number of monitoring points")
    p.add_argument("--drones", type=int, default=6)
    p.add_argument("--cell-size", type=float, default=1000.0)
    p.add_argument("--workers", type=int, default=1,
                   help=">1 runs trials in parallel; runtime_s is then only comparable within the same run")
    p.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    p.add_argument("--fire-id", type=int, default=3885)
    p.add_argument("--year", type=int, default=2018)
    p.add_argument("--snapshots", type=int, default=8)
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--stages", type=int, nargs="+", default=[6, 40, 80],
                   help="feds: mission stages whose planning state is used as the base scenario")
    p.add_argument("--params", default=None,
                   help="JSON of ACO parameters (e.g. outputs/tuning/defaults_params.json); "
                        "--ants and --iterations still set the budget")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _load_params(path):
    if not path:
        return None
    from tuning import load_params
    return load_params(path)


def main():
    a = parse_args()
    cfg = ExperimentConfig(phase=a.phase, scenario=a.source, fractions=tuple(a.fractions),
                           localities=tuple(a.localities), kind=a.kind, direction=a.direction, trials=a.trials, first_seed=a.first_seed,
                           warmup=a.warmup, ants=a.ants, iterations=a.iterations, vns_evaluations=a.vns_evals,
                           n_points=a.points, drones=a.drones, cell_size_m=a.cell_size, workers=a.workers,
                           params=_load_params(a.params), selective_radius_m=a.selective_radius,
                           planners=tuple(a.planners))
    print(f"budget per replan: ACO {cfg.ants * cfg.iterations} evaluations, VNS {cfg.vns_params().max_evaluations}")
    if a.source == "synthetic":
        rows, traces = run_synthetic(cfg)
    else:
        states = captured_states(a.gpkg, a.fire_id, a.year, snapshots=a.snapshots, cache=a.cache,
                                 cell_size_m=a.cell_size, drones=a.drones, stages=tuple(a.stages))
        bases = {f"fire{a.fire_id}_stage{k}": s for k, s in sorted(states.items())}
        rows, traces = run_from_states(bases, cfg)
    write_csv(a.out, RESULT_FIELDS, rows)
    stem = a.out[:-4] if a.out.endswith(".csv") else a.out
    write_csv(f"{stem}_trace.csv", TRACE_FIELDS, traces)
    print(f"wrote {a.out} ({len(rows)} rows) and {stem}_trace.csv ({len(traces)} rows)")


if __name__ == "__main__":
    main()
