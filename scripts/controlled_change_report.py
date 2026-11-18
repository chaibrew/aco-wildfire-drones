import argparse
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment.stats import (_p_values, _t_ci, change_interaction, holm_adjust, load_rows,  # noqa: E402
                              load_trace, paired_summary, recovery_summary, reset_share_summary)

METRICS = ("margin", "total_overdue", "overdue_count")
REF = "aco_full_reset"
STAGES = ("fire3885_stage6", "fire3885_stage40", "fire3885_stage80")


def path(d, name):
    return os.path.join(d, name)


def fmt(x, digits=3):
    return f"{x:+.{digits}f}"


def pair_key(r):
    return (r["scenario"], r["trial"], r["target_fraction"], r["locality"])


def diff_stats(a_rows, b_rows, metric, a_planner, b_planner):
    a = {pair_key(r): r[metric] for r in a_rows if r["planner"] == a_planner}
    b = {pair_key(r): r[metric] for r in b_rows if r["planner"] == b_planner}
    by_cluster = defaultdict(list)
    for k in a.keys() & b.keys():
        by_cluster[k[:2]].append(a[k] - b[k])
    diffs = [statistics.mean(v) for v in by_cluster.values()]
    lo, hi = _t_ci(diffs)
    t_p, _ = _p_values(diffs)
    ref_mean = statistics.mean(b[k] for k in a.keys() & b.keys())
    return {"mean": statistics.mean(diffs), "lo": lo, "hi": hi, "p": t_p, "n": len(diffs),
            "better": sum(d < 0 for d in diffs), "ref_mean": ref_mean, "pairs": len(a.keys() & b.keys())}


def line(label, s, rel=False):
    pct = f" ({100 * s['mean'] / s['ref_mean']:+.2f}%)" if rel else ""
    return (f"  {label:34s} {fmt(s['mean'])}{pct} [{fmt(s['lo'])}, {fmt(s['hi'])}]  better {s['better']}/{s['n']}"
            f"  p {s['p']:.3g}")


def main_sweep(rows, title):
    print(f"\n## {title}: every planner minus full reset (final plans)")
    for m in METRICS:
        summary = [r for r in paired_summary(rows, m, REF, "all")]
        ref_mean = next(r["mean"] for r in summary if r["planner"] == REF)
        print(f"\n{m} (full reset mean {ref_mean:.4f})")
        for r in summary:
            if "mean_diff" not in r:
                continue
            print(f"  {r['planner']:22s} {fmt(r['mean_diff'], 4)} ({100 * r['mean_diff'] / ref_mean:+.2f}%) "
                  f"[{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}]  better {r['better']}/{r['n_clusters']}  "
                  f"Holm p {r['t_p_holm']:.3g}")


def recovery(trace_path, title, values=("best_margin", "best_total_overdue"), reference=REF, planners=None):
    if not os.path.exists(trace_path):
        return
    trace = load_trace(trace_path)
    if planners:
        trace = [r for r in trace if r["planner"] in set(planners) | {reference}]
    for v in values:
        print(f"\n## Recovery, {title}: {v} minus {reference}'s, after k evaluations")
        for r in recovery_summary(trace, v, reference, (20, 60, 120, 300, 600, 1200)):
            if "mean_diff" in r:
                print(f"  {r['evaluations']:>6g} {r['planner']:22s} {fmt(r['mean_diff'], 4)} "
                      f"[{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}]  better {r['better']}/{r['n_clusters']}  "
                      f"Holm p {r['t_p_holm']:.3g}")


def interactions(rows, title):
    print(f"\n## Interaction, {title} (negative: selective reset relatively better under the first condition)")
    small = [r for r in rows if r["target_fraction"] <= 0.20]
    for m in METRICS:
        tests = []
        for ref in ("aco_retention", REF):
            tests.append(("locality", ref, change_interaction(small, m, "aco_selective_reset", ref, "locality")))
            tests.append(("size", ref, change_interaction(rows, m, "aco_selective_reset", ref, "fraction", 0.5)))
        for (factor, ref, t), adj in zip(tests, holm_adjust([t["t_p"] for _, _, t in tests])):
            print(f"  {m:14s} {factor:8s} vs {ref:15s} {fmt(t['mean_interaction'], 4)} "
                  f"[{fmt(t['ci_lo'], 4)}, {fmt(t['ci_hi'], 4)}]  p {t['t_p']:.3g}  Holm p {adj:.3g}")
    for m in METRICS:
        for ref in ("aco_retention", REF):
            t = change_interaction(rows, m, "aco_selective_reset", ref, "fraction", 0.15)
            print(f"  {m:14s} size@0.15 vs {ref:15s} {fmt(t['mean_interaction'], 4)} "
                  f"[{fmt(t['ci_lo'], 4)}, {fmt(t['ci_hi'], 4)}]  p {t['t_p']:.3g}  (exploratory, no adjustment)")


def reset_share(rows, title):
    print(f"\n## Selective reset by share of points reset, {title}")
    for m in METRICS:
        for r in reset_share_summary(rows, m):
            print(f"  {m:14s} [{r['share_lo']:.2f},{r['share_hi']:.2f}) vs {r['reference']:15s} "
                  f"{fmt(r['mean_diff'], 4)} [{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}]  trials {r['n_clusters']}")


def mean_share(rows):
    shares = [float(r["reset_points"]) / max(1.0, float(r["n_after"])) for r in rows
              if r["planner"] == "aco_selective_reset" and r.get("reset_points") not in ("", None)]
    return statistics.mean(shares) if shares else float("nan")


def main():
    p = argparse.ArgumentParser(description="Print every number in docs/controlled-change-results.md.")
    p.add_argument("--dir", default="outputs/cc")
    d = p.parse_args().dir
    camp = load_rows(path(d, "camp.csv"))
    syn = load_rows(path(d, "synthetic.csv"))

    main_sweep(camp, "Camp Fire, tuned")
    main_sweep(syn, "Synthetic, tuned")
    recovery(path(d, "camp_trace.csv"), "Camp Fire, tuned")
    recovery(path(d, "synthetic_trace.csv"), "Synthetic, tuned")
    for name, title in (("camp", "Camp Fire, tuned"), ("synthetic", "Synthetic, tuned"),
                        ("camp_low_rho", "Camp Fire, low rho")):
        recovery(path(d, f"{name}_trace.csv"), title, ("best_margin",), "aco_retention", ("aco_selective_reset",))

    low = load_rows(path(d, "camp_low_rho.csv"))
    main_sweep(low, "Camp Fire, low rho")
    recovery(path(d, "camp_low_rho_trace.csv"), "Camp Fire, low rho")
    print("\n## Low rho minus tuned, same pairs")
    for planner in ("aco_retention", REF, "aco_fast_evaporation", "aco_selective_reset"):
        for m in METRICS:
            print(line(f"{planner} {m}", diff_stats(low, camp, m, planner, planner)))

    r1000 = load_rows(path(d, "camp_selective_r1000.csv"))
    combined = [r for r in camp if r["planner"] != "aco_selective_reset"] + r1000
    interactions(camp, "Camp Fire, tuned")
    interactions(syn, "Synthetic, tuned")
    interactions(low, "Camp Fire, low rho")
    interactions(combined, "Camp Fire, 1000 m radius")
    reset_share(camp, "Camp Fire, tuned")
    print(f"\n## 1000 m radius: mean share of points reset {mean_share(r1000):.3f} (0 m: {mean_share(camp):.3f})")
    for m in METRICS:
        print(line(f"r1000 - r0 {m}", diff_stats(r1000, camp, m, "aco_selective_reset", "aco_selective_reset")))
        print(line(f"r1000 - retention {m}", diff_stats(r1000, camp, m, "aco_selective_reset", "aco_retention")))
        print(line(f"r1000 - full reset {m}", diff_stats(r1000, camp, m, "aco_selective_reset", REF)))

    print("\n## Budget (seeds 5000-5009): planner at budget B minus ACO full reset at 1200 evaluations")
    first10 = [r for r in camp if int(r["trial"]) < 5010]
    budget_files = [("vns", 1200, first10, "vns")] + [("vns", e, None, f"camp_vns{e}.csv") for e in (4800, 12000)]
    budget_files += [(pl, 1200, first10, pl) for pl in ("aco_full_reset", "aco_retention")]
    budget_files += [(pl, e, None, f"camp_aco{e}.csv") for e in (4800, 12000) for pl in (REF, "aco_retention")]
    loaded = {}
    for planner, evals, rows, src in budget_files:
        if rows is None:
            if not os.path.exists(path(d, src)):
                continue
            rows = load_rows(path(d, src))
        loaded[(planner, evals)] = rows
        if (planner, evals) != (REF, 1200):
            for m in METRICS:
                print(line(f"{planner} @ {evals} {m}", diff_stats(rows, first10, m, planner, REF), rel=m == "margin"))
        means = {m: statistics.mean(float(r[m]) for r in rows if r["planner"] == planner) for m in METRICS}
        print(f"  {planner} @ {evals} mean " + "  ".join(f"{m} {v:.3f}" for m, v in means.items()))
        times = [r["runtime_s"] for r in rows if r["planner"] == planner]
        print(f"  {planner} @ {evals} seconds per replan (parallel run) {statistics.mean(times):.2f}")
    print("\n## Budget: VNS minus ACO full reset at the same budget")
    for e in (1200, 4800, 12000):
        if ("vns", e) in loaded and (REF, e) in loaded:
            for m in METRICS:
                print(line(f"vns - aco_full_reset @ {e} {m}", diff_stats(loaded[("vns", e)], loaded[(REF, e)], m,
                                                                         "vns", REF), rel=m == "margin"))
    print("\n## Budget: ACO at budget B minus ACO at 1200, same strategy")
    for e in (4800, 12000):
        for pl in (REF, "aco_retention"):
            if (pl, e) in loaded:
                for m in METRICS:
                    print(line(f"{pl} @ {e} - @ 1200 {m}", diff_stats(loaded[(pl, e)], first10, m, pl, pl)))
    print("\n## Budget: retention minus full reset at each ACO budget")
    for e in (1200, 4800, 12000):
        if ("aco_retention", e) in loaded and (REF, e) in loaded:
            for m in METRICS:
                print(line(f"retention - full reset @ {e} {m}",
                           diff_stats(loaded[("aco_retention", e)], loaded[(REF, e)], m, "aco_retention", REF)))

    print("\n## Stage-40 diagnostic: retention / selective reset minus full reset, by stage (seeds 5000-5009)")
    settings = [("untuned, 10x15, cycle", "camp_untuned_cycle.csv"), ("tuned, 20x60, cycle", "camp_cycle.csv"),
                ("tuned, 20x60, up (main)", None)]
    for label, src in settings:
        rows = first10 if src is None else load_rows(path(d, src))
        for stage in STAGES:
            sr = [r for r in rows if r["scenario"] == stage]
            for pl in ("aco_retention", "aco_selective_reset"):
                for m in ("margin", "total_overdue"):
                    s = diff_stats(sr, sr, m, pl, REF)
                    print(f"  {label:26s} {stage:18s} {pl:20s} {m:14s} {fmt(s['mean'])} "
                          f"({100 * s['mean'] / s['ref_mean']:+.1f}%) better {s['better']}/{s['n']}")
    print("\n## Tuned minus untuned (cycling rule, same changes)")
    tuned, untuned = load_rows(path(d, "camp_cycle.csv")), load_rows(path(d, "camp_untuned_cycle.csv"))
    for pl in (REF, "aco_retention", "aco_selective_reset"):
        for m in METRICS:
            print(line(f"{pl} {m}", diff_stats(tuned, untuned, m, pl, pl), rel=True))

    print("\n## Dispersion of concentrated vs distributed changes (fractions <= 0.20), Camp Fire")
    disp = defaultdict(list)
    for r in camp:
        if r["planner"] == REF and r["target_fraction"] <= 0.20:
            disp[(r["locality"], r["target_fraction"])].append(float(r["dispersion"]))
    for k in sorted(disp):
        print(f"  {k[0]:13s} {k[1]:.2f} {statistics.mean(disp[k]):.3f}")


if __name__ == "__main__":
    main()
