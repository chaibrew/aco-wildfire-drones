import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

OUT = sys.argv[1]
MEASURES = ("true_overdue_mean", "true_J_mean", "true_max_gap_mean", "map_overdue_mean", "planning_s_mean")
rows = list(csv.DictReader(open(f"{OUT}/missions.csv")))
by = defaultdict(dict)
for r in rows:
    by[(int(r["drones"]), r["strategy"], int(r["seed"]))][r["setting"]] = r

summary = []
for drones in sorted({k[0] for k in by}):
    for strategy in sorted({k[1] for k in by}) + ["all"]:
        keys = [k for k in by if k[0] == drones and (strategy == "all" or k[1] == strategy) and len(by[k]) == 2]
        for m in MEASURES:
            u = np.array([float(by[k]["untuned"][m]) for k in keys])
            t = np.array([float(by[k]["tuned"][m]) for k in keys])
            d = t - u
            p = wilcoxon(t, u).pvalue if np.any(d != 0) else 1.0
            summary.append({"drones": drones, "strategy": strategy, "measure": m, "n": len(keys),
                            "untuned": u.mean(), "tuned": t.mean(), "mean_change": d.mean(),
                            "pct_change": 100 * d.mean() / u.mean() if u.mean() else 0.0,
                            "tuned_better": int((d < 0).sum()), "tied": int((d == 0).sum()),
                            "tuned_worse": int((d > 0).sum()), "p": p})

with open(f"{OUT}/missions_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0]))
    w.writeheader()
    w.writerows(summary)

for m in MEASURES:
    print(f"\n{m}  (lower is better; b/t/w = tuned better/tied/worse)")
    print(f"{'drones':>6s} {'strategy':>17s} {'n':>3s} {'untuned':>9s} {'tuned':>9s} {'change':>8s} {'b/t/w':>8s} {'p':>8s}")
    for s in summary:
        if s["measure"] == m:
            print(f"{s['drones']:6d} {s['strategy']:>17s} {s['n']:3d} {s['untuned']:9.3f} {s['tuned']:9.3f} "
                  f"{s['pct_change']:+7.1f}% {s['tuned_better']:2d}/{s['tied']}/{s['tuned_worse']:<2d} {s['p']:8.2g}")
