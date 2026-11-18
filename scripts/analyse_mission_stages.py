import argparse
import csv
import math
import statistics
from collections import defaultdict

METRICS = [
    ("map_objective", True),
    ("true_objective", True),
    ("true_overdue_count", True),
    ("perimeter_error_m", True),
    ("growth_found_fraction", False),
    ("front_on_map_fraction", False),
    ("map_points", None),
    ("true_front_points", None),
]


def main():
    p = argparse.ArgumentParser(description="Compare planners on the replan rows of a mission replay, before and after a time split.")
    p.add_argument("stages")
    p.add_argument("--baseline", default="greedy")
    p.add_argument("--split", type=float, default=12.0, help="hours into the mission where the windows split")
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.stages)))
    start = min(float(r["time_s"]) for r in rows)
    groups = defaultdict(list)
    for r in rows:
        hours = (float(r["time_s"]) - start) / 3600
        window = f"hours 0-{a.split:g}" if hours < a.split else f"hours {a.split:g}-end"
        groups[(window, r["planner"], int(r["seed"]))].append(r)
    planners = sorted({r["planner"] for r in rows})
    seeds = sorted({int(r["seed"]) for r in rows})
    for window in sorted({w for w, _, _ in groups}):
        print(f"== {window}, difference from {a.baseline} over {len(seeds)} seeds")
        for key, lower in METRICS:
            means = {pl: {s: statistics.mean(float(r[key]) for r in groups[(window, pl, s)]) for s in seeds}
                     for pl in planners}
            print(f"  {key}")
            for pl in planners:
                mean = statistics.mean(means[pl].values())
                if pl == a.baseline:
                    print(f"    {pl:22s} {mean:9.3f}")
                    continue
                d = [means[pl][s] - means[a.baseline][s] for s in seeds]
                sd = statistics.stdev(d) if len(d) > 1 else 0.0
                t = statistics.mean(d) / (sd / math.sqrt(len(d))) if sd > 0 else float("nan")
                better = "" if lower is None else f"  better {sum((x < 0) if lower else (x > 0) for x in d)}/{len(d)}"
                print(f"    {pl:22s} {mean:9.3f} {statistics.mean(d):+9.3f}  t {t:+6.2f}{better}")
        print()


if __name__ == "__main__":
    main()
