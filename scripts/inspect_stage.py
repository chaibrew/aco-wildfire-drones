import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment.replay import load_stages, scan_anomalies, stage_report  # noqa: E402


def share(x):
    return "n/a" if x != x else f"{100 * x:.0f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stages_csv")
    p.add_argument("--stage", type=int)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--metric", default="true_objective")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--threshold", type=float, default=4.0)
    a = p.parse_args()
    rows = load_stages(a.stages_csv)
    if a.stage is not None:
        print(f"stage {a.stage} vs stages within +-{a.window} ({a.metric}, medians across seeds)\n")
        for r in stage_report(rows, a.stage, a.window, a.metric):
            print(f"drones {r['drones']:2d} {r['planner']:8s} at stage {r['median_at_stage']:.3f}  neighbours "
                  f"{r['median_neighbours']:.3f}  diff {r['difference']:+.3f}  | seeds peaking here "
                  f"{r['seeds_peaking']}/{r['seeds']}  top seed {r['top_seed']} ({r['top_seed_value']:.3f}, "
                  f"{share(r['top_seed_share_of_sum'])} of the sum)  snapshots {r['seeds_with_snapshot']}/"
                  f"{r['seeds']}  points {r['median_map_points']:.0f}  affected {r['median_affected_fraction']:.2f}")
    if a.scan:
        print(f"stages whose median {a.metric} jumps from the previous stage (robust z >= {a.threshold}):\n")
        found = scan_anomalies(rows, a.metric, a.threshold)
        for r in found:
            print(f"drones {r['drones']:2d} {r['planner']:8s} stage {r['stage']:3d}  jump {r['jump']:+.3f}  z {r['robust_z']:.1f}")
        if not found:
            print("none")
    if a.stage is None and not a.scan:
        p.error("give --stage N and/or --scan")


if __name__ == "__main__":
    main()