import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment.stats import (GROUPINGS, METRICS, change_interaction, format_interactions,  # noqa: E402
                              format_recovery, format_summary, load_rows, load_trace, paired_summary,
                              recovery_summary, reset_share_summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--metric", nargs="+", default=["margin", "total_overdue", "overdue_count"], choices=METRICS)
    p.add_argument("--reference", default="aco_full_reset", help="planner the others are compared with")
    p.add_argument("--group", default="cell", choices=sorted(GROUPINGS))
    p.add_argument("--include-pilot", action="store_true")
    p.add_argument("--split", type=float, default=0.5, help="fraction cut-off between small and large changes")
    p.add_argument("--no-interaction", action="store_true")
    p.add_argument("--trace", help="trace csv (default: <csv stem>_trace.csv if it exists)")
    p.add_argument("--recovery-value", default="best_margin",
                   choices=("best_total_overdue", "best_margin", "best_objective"))
    p.add_argument("--checkpoints", type=float, nargs="+", default=[20, 60, 120, 300, 600, 1200])
    a = p.parse_args()

    rows = load_rows(a.csv, include_pilot=a.include_pilot)
    if not rows:
        sys.exit("no rows (are they all pilot rows? use --include-pilot)")
    trials = {(r["scenario"], r["trial"]) for r in rows}
    print(f"{len(rows)} rows, {len(trials)} independent trial clusters\n")
    for m in a.metric:
        print(format_summary(paired_summary(rows, m, a.reference, a.group), m, a.reference, a.group), "\n")

    if not a.no_interaction and "aco_selective_reset" in {r["planner"] for r in rows}:
        others = sorted({r["planner"] for r in rows if r["planner"].startswith("aco_")} - {"aco_selective_reset"})
        results = [change_interaction(rows, m, "aco_selective_reset", ref, factor, a.split)
                   for m in a.metric for ref in others for factor in ("locality", "fraction")]
        print("Does selective reset do relatively better after concentrated / small changes?\n")
        print(format_interactions(results), "\n")

    if any(r.get("reset_points") not in ("", None) for r in rows):
        print("Selective reset by the share of points it actually reset (diff = selective - reference, "
              f"{a.metric[0]}, averaged within each trial)\n")
        for r in reset_share_summary(rows, a.metric[0]):
            print(f"  reset share [{r['share_lo']:.2f},{r['share_hi']:.2f})  vs {r['reference']:16s} "
                  f"{r['mean_diff']:+8.3f}  95% CI [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  "
                  f"trials {r['n_clusters']}  pairs {r['n_pairs']}")
        print()

    trace_path = a.trace or (a.csv[:-4] if a.csv.endswith(".csv") else a.csv) + "_trace.csv"
    if os.path.exists(trace_path):
        summary = recovery_summary(load_trace(trace_path), a.recovery_value, a.reference, tuple(a.checkpoints),
                                   include_pilot=a.include_pilot)
        print(format_recovery(summary, a.recovery_value, a.reference))


if __name__ == "__main__":
    main()
