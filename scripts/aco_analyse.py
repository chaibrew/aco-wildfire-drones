import csv
import math
import statistics
import sys

METRICS = [
    ("map_J_mean", "Map J (mean over stages)"),
    ("map_overdue_mean", "Map overdue points (mean)"),
    ("true_J_mean", "True-front J (mean)"),
    ("true_overdue_mean", "True-front overdue points (mean)"),
    ("map_J_max", "Map J (worst stage)"),
]

rows = list(csv.DictReader(open(sys.argv[1])))
by: dict[str, dict[int, dict]] = {}
for r in rows:
    by.setdefault(r["planner"], {})[int(r["seed"])] = r
base = by["smoke_router"]
seeds = sorted(base)
print(f"{len(seeds)} seeds\n")
for key, label in METRICS:
    print(label)
    print(f"  {'planner':22s} {'mean':>9s} {'diff vs smoke':>14s} {'t':>7s} {'lower on':>9s}")
    for name, d in by.items():
        vals = [float(d[s][key]) for s in seeds]
        if name == "smoke_router":
            print(f"  {name:22s} {statistics.mean(vals):9.3f}")
            continue
        diffs = [float(d[s][key]) - float(base[s][key]) for s in seeds]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        t = m / (sd / math.sqrt(len(diffs))) if sd > 0 else float("nan")
        wins = sum(x < 0 for x in diffs)
        print(f"  {name:22s} {statistics.mean(vals):9.3f} {m:+14.3f} {t:+7.2f} {wins:>5d}/{len(seeds)}")
    print()
plan = {name: statistics.mean(float(d[s]["planning_s_mean"]) for s in seeds) for name, d in by.items()}
print("Mean planning time per replan (s): " + ", ".join(f"{k} {v:.3f}" for k, v in plan.items()))
