import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

OUT = sys.argv[1]
MEASURES = ("true_overdue_mean", "true_J_mean", "true_max_gap_mean", "map_overdue_mean", "planning_s_mean")
rows = list(csv.DictReader(open(f"{OUT}/missions.csv")))
runs = defaultdict(dict)
for r in rows:
    name = r["strategy"] if r["setting"] == "baseline" else f"aco_{r['strategy']}_{r['setting']}"
    runs[(int(r["drones"]), name)][int(r["seed"])] = r


def paired(drones, pairs, m, label, ref_name, other_name):
    """pairs: (reference planner, other planner) pairs whose seeds are matched and pooled."""
    matched = [(runs[(drones, a)][s], runs[(drones, b)][s]) for a, b in pairs
               if (drones, a) in runs and (drones, b) in runs for s in sorted(set(runs[(drones, a)]) & set(runs[(drones, b)]))]
    if not matched:
        return None
    u = np.array([float(a[m]) for a, _ in matched])
    t = np.array([float(b[m]) for _, b in matched])
    d = t - u
    return {"drones": drones, "comparison": label, "reference": ref_name, "other": other_name, "measure": m,
            "n": len(matched), "reference_mean": u.mean(), "other_mean": t.mean(), "mean_change": d.mean(),
            "pct_change": 100 * d.mean() / u.mean() if u.mean() else 0.0,
            "other_better": int((d < 0).sum()), "tied": int((d == 0).sum()), "other_worse": int((d > 0).sum()),
            "p": wilcoxon(t, u).pvalue if np.any(d != 0) else 1.0}


strategies = sorted({r["strategy"] for r in rows if r["setting"] != "baseline"})
baselines = sorted({r["strategy"] for r in rows if r["setting"] == "baseline"})
summary = []
for drones in sorted({k[0] for k in runs}):
    for m in MEASURES:
        for s in strategies:
            summary.append(paired(drones, [(f"aco_{s}_untuned", f"aco_{s}_tuned")], m, "tuned_vs_untuned",
                                  f"aco_{s}_untuned", f"aco_{s}_tuned"))
        summary.append(paired(drones, [(f"aco_{s}_untuned", f"aco_{s}_tuned") for s in strategies], m,
                              "tuned_vs_untuned", "aco_untuned_all", "aco_tuned_all"))
        for setting in ("tuned", "untuned"):
            for s in strategies:
                if s != "full_reset":
                    summary.append(paired(drones, [(f"aco_full_reset_{setting}", f"aco_{s}_{setting}")], m,
                                          f"{setting}_vs_full_reset", f"aco_full_reset_{setting}", f"aco_{s}_{setting}"))
        for b in baselines:
            for setting in ("tuned", "untuned"):
                for s in strategies:
                    summary.append(paired(drones, [(b, f"aco_{s}_{setting}")], m, f"aco_{setting}_vs_{b}",
                                          b, f"aco_{s}_{setting}"))
summary = [s for s in summary if s]

with open(f"{OUT}/missions_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

print("planner means (lower is better)")
for (drones, name), by_seed in sorted(runs.items()):
    v = [float(r["true_overdue_mean"]) for r in by_seed.values()]
    j = [float(r["true_J_mean"]) for r in by_seed.values()]
    t = [float(r["planning_s_mean"]) for r in by_seed.values()]
    print(f"  d{drones:<3d} {name:30s} n={len(v):2d}  overdue {np.mean(v):7.2f}  true J {np.mean(j):7.3f}  "
          f"planning {np.mean(t):6.3f} s")
for m in ("true_overdue_mean", "true_J_mean"):
    print(f"\n{m}  (b/t/w = other better/tied/worse than reference)")
    print(f"{'d':>3s} {'reference':>18s} {'other':>30s} {'n':>3s} {'ref':>8s} {'other':>8s} {'change':>8s} {'b/t/w':>8s} {'p':>8s}")
    for s in summary:
        if s["measure"] == m:
            print(f"{s['drones']:3d} {s['reference']:>18s} {s['other']:>30s} {s['n']:3d} {s['reference_mean']:8.3f} "
                  f"{s['other_mean']:8.3f} {s['pct_change']:+7.1f}% {s['other_better']:2d}/{s['tied']}/{s['other_worse']:<2d} {s['p']:8.2g}")
