import csv
import math
import statistics
import sys

METRICS = [
    ("map_J_mean", "Map J (mean over stages)", True),
    ("map_overdue_mean", "Map overdue points (mean)", True),
    ("true_J_mean", "True-front J (mean)", True),
    ("true_overdue_mean", "True-front overdue points (mean)", True),
    ("map_J_max", "Map J (worst stage)", True),
    ("perimeter_error_mean_m", "Map perimeter error, m (mean over stages)", True),
    ("perimeter_error_p95_mean_m", "Map perimeter error, 95th percentile, m", True),
    ("growth_found_mean", "Growth since last snapshot found by the map (fraction)", False),
    ("known_burned_mean", "Burned area known (fraction)", False),
    ("front_on_map_mean", "True front cells on the map (fraction)", False),
]

rows = list(csv.DictReader(open(sys.argv[1])))
by: dict[str, dict[int, dict]] = {}
for r in rows:
    by.setdefault(r["planner"], {})[int(r["seed"])] = r
baseline = sys.argv[2] if len(sys.argv) > 2 else "greedy"  # e.g. smoke_router, vns, aco_full_reset
base = by[baseline]
diff_label = f"diff vs {baseline}"
seeds = sorted(base)
print(f"{len(seeds)} seeds\n")
for key, label, lower in METRICS:
    if key not in rows[0]:
        continue
    print(label)
    print(f"  {'planner':22s} {'mean':>9s} {diff_label:>14s} {'t':>7s} {'better on':>10s}")
    for name, d in by.items():
        vals = [float(d[s][key]) for s in seeds]
        if name == baseline:
            print(f"  {name:22s} {statistics.mean(vals):9.3f}")
            continue
        diffs = [float(d[s][key]) - float(base[s][key]) for s in seeds]
        m = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
        t = m / (sd / math.sqrt(len(diffs))) if sd > 0 else float("nan")
        wins = sum((x < 0) if lower else (x > 0) for x in diffs)
        print(f"  {name:22s} {statistics.mean(vals):9.3f} {m:+14.3f} {t:+7.2f} {wins:>6d}/{len(seeds)}")
    print()
plan = {name: statistics.mean(float(d[s]["planning_s_mean"]) for s in seeds) for name, d in by.items()}
print("Mean planning time per replan (s): " + ", ".join(f"{k} {v:.3f}" for k, v in plan.items()))
