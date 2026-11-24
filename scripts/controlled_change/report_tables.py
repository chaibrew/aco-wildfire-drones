import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from experiment.stats import change_interaction, holm_adjust, load_rows, load_trace, paired_summary  # noqa: E402

REF = "aco_full_reset"
STRATEGIES = ("aco_retention", "aco_selective_reset")
METRICS = ("margin", "total_overdue", "overdue_count")
STAGES = ("fire3885_stage6", "fire3885_stage40", "fire3885_stage80")


def rows_at(trace, evaluations):
    runs = defaultdict(list)
    for r in trace:
        key = (r["planner"], r["scenario"], r["trial"], float(r["target_fraction"]), r["locality"])
        runs[key].append((float(r["evaluations"]), float(r["best_margin"]), float(r["best_total_overdue"])))
    out = []
    for (planner, scenario, trial, fraction, locality), points in runs.items():
        best = [p for p in sorted(points) if p[0] <= evaluations]
        if best:
            out.append({"planner": planner, "scenario": scenario, "trial": trial, "target_fraction": fraction,
                        "locality": locality, "margin": best[-1][1], "total_overdue": best[-1][2]})
    return out


def diffs(rows, metric, grouping="all"):
    return [r for r in paired_summary(rows, metric, REF, grouping) if "mean_diff" in r]


def show(label, r, holm=None):
    tail = f"  Holm p {holm:.3g}" if holm is not None else ""
    print(f"  {label:44s} {r['mean_diff']:+8.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
          f"better {r['better']} worse {r['worse']} of {r['n_clusters']}  p {r['t_p']:.3g}{tail}")


def final_by_size(rows):
    print("\nFinal plans, strategy - full reset, by change size (Holm over 30 tests)")
    tests = [(m, r) for m in METRICS for r in diffs(rows, m, "fraction") if r["planner"] in STRATEGIES]
    for (m, r), adj in zip(tests, holm_adjust([r["t_p"] for _, r in tests])):
        show(f"{m} {r['planner']} {r['target_fraction']:.2f}", r, adj)


def baselines(rows, base):
    both = [r for r in rows if r["planner"] == REF] + base
    print("\nBaselines - full reset, final plans (Holm over 6 tests)")
    for m in METRICS:
        means = {p: statistics.mean(r[m] for r in rows + base if r["planner"] == p)
                 for p in (REF, *STRATEGIES, "vns", "greedy")}
        print(f"  {m} means " + "  ".join(f"{p} {v:.3f}" for p, v in means.items()))
    tests = [(m, r) for m in METRICS for r in diffs(both, m)]
    for (m, r), adj in zip(tests, holm_adjust([r["t_p"] for _, r in tests])):
        show(f"{m} {r['planner']}", r, adj)
    for stage in STAGES:
        sub = [r for r in both if r["scenario"] == stage]
        show(f"total_overdue vns {stage}", next(r for r in diffs(sub, "total_overdue") if r["planner"] == "vns"))


def recovery(trace):
    print("\nTotal overdue of the best plan so far, strategy - full reset, by evaluations")
    for k in range(20, 1201, 20):
        rows = rows_at(trace, k)
        full = statistics.mean(r["total_overdue"] for r in rows if r["planner"] == REF)
        for r in diffs(rows, "total_overdue"):
            if r["planner"] in STRATEGIES:
                show(f"{k:5d} {r['planner']} (full reset mean {full:.3f})", r)
    for k in (20, 100):
        rows = rows_at(trace, k)
        print(f"  retention mean at {k}: {statistics.mean(r['total_overdue'] for r in rows if r['planner'] == 'aco_retention'):.3f}")


def breakdowns(rows, label):
    print(f"\n{label}: total overdue, strategy - full reset")
    for r in diffs(rows, "total_overdue", "fraction"):
        if r["planner"] in STRATEGIES:
            show(f"size {r['target_fraction']:.2f} {r['planner']}", r)
    small = [r for r in rows if r["target_fraction"] <= 0.20]
    for r in diffs(small, "total_overdue", "locality"):
        if r["planner"] in STRATEGIES:
            show(f"<=20% {r['locality']} {r['planner']}", r)
    for stage in STAGES:
        sub = [r for r in rows if r["scenario"] == stage]
        full = statistics.mean(r["total_overdue"] for r in sub if r["planner"] == REF)
        for r in diffs(sub, "total_overdue"):
            if r["planner"] in STRATEGIES:
                show(f"{stage} {r['planner']} (full reset {full:.2f})", r)


def interactions(rows, metrics, label):
    print(f"\n{label}: selective reset gain, concentrated - distributed (<=20%) and small - large (split 0.5)")
    small = [r for r in rows if r["target_fraction"] <= 0.20]
    for m in metrics:
        tests = []
        for ref in ("aco_retention", REF):
            tests.append(("locality", change_interaction(small, m, "aco_selective_reset", ref, "locality")))
            tests.append(("size", change_interaction(rows, m, "aco_selective_reset", ref, "fraction", 0.5)))
        for (factor, t), adj in zip(tests, holm_adjust([t["t_p"] for _, t in tests])):
            print(f"  {m:14s} {factor:8s} vs {t['reference']:15s} {t['mean_interaction']:+.3f} "
                  f"[{t['ci_lo']:+.3f}, {t['ci_hi']:+.3f}]  p {t['t_p']:.3g}  Holm p {adj:.3g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("baselines")
    a = p.parse_args()
    rows = load_rows(a.csv)
    trace = load_trace(a.csv[:-4] + "_trace.csv")
    final_by_size(rows)
    baselines(rows, load_rows(a.baselines))
    recovery(trace)
    breakdowns(rows_at(trace, 20), "After 20 evaluations")
    breakdowns(rows, "Final plans")
    interactions(rows, METRICS, "Final plans")
    interactions(rows_at(trace, 20), ("margin", "total_overdue"), "After 20 evaluations")


if __name__ == "__main__":
    main()
