"""Paired analysis of scripts/run_change_experiment.py output.

  python scripts/analyse_change_experiment.py outputs/change_synthetic.csv
  python scripts/analyse_change_experiment.py outputs/change_synthetic.csv --metric total_overdue \
      --reference aco_retention --group locality
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment.stats import (GROUPINGS, METRICS, change_interaction, format_interactions, format_summary,  # noqa: E402
                              load_rows, load_trace, paired_summary, trace_summary)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--metric", nargs="+", default=["margin", "total_overdue", "overdue_count"], choices=METRICS)
    p.add_argument("--reference", default="aco_full_reset", help="planner the others are compared with")
    p.add_argument("--group", default="cell", choices=sorted(GROUPINGS))
    p.add_argument("--include-pilot", action="store_true")
    p.add_argument("--split", type=float, default=0.5, help="fraction cut-off between small and large changes")
    p.add_argument("--no-interaction", action="store_true")
    p.add_argument("--trace", help="trace csv (default: <csv stem>_trace.csv if it exists)")
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

    trace_path = a.trace or (a.csv[:-4] if a.csv.endswith(".csv") else a.csv) + "_trace.csv"
    if os.path.exists(trace_path):
        print("median best margin after a change, by evaluations used")
        for r in trace_summary(load_trace(trace_path)):
            print(f"  {r['planner']:22s} evals={r['evaluations']:<5g} {r['median_best_margin']:8.3f}  (n={r['n']})")


if __name__ == "__main__":
    main()
