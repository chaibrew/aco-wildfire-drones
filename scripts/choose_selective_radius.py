import os
import re
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from experiment.stats import _t_ci, load_rows  # noqa: E402

METRIC = "total_overdue"


def main(paths):
    values = defaultdict(dict)
    for path in (p for p in paths if not p.endswith("_trace.csv")):
        radius = float(re.search(r"_r(\d+(?:\.\d+)?)\.csv$", path).group(1))
        for r in load_rows(path, include_pilot=True):
            if r["planner"] == "aco_selective_reset":
                values[radius][(r["scenario"], r["trial"], r["target_fraction"], r["locality"])] = r[METRIC]
    radii = sorted(values)
    means = {rad: statistics.mean(values[rad].values()) for rad in radii}
    base = values[radii[0]]
    print(f"selective reset, {METRIC} (pilot rows, lower is better)\n")
    print(f"{'radius m':>9s} {'pairs':>6s} {'mean':>9s} {'diff vs ' + format(radii[0], 'g'):>11s} {'95% CI':>19s}")
    for rad in radii:
        by_cluster = defaultdict(list)
        for pair, v in values[rad].items():
            if pair in base:
                by_cluster[pair[:2]].append(v - base[pair])
        diffs = [statistics.mean(d) for d in by_cluster.values()]
        lo, hi = _t_ci(diffs)
        print(f"{rad:9g} {len(values[rad]):6d} {means[rad]:9.3f} {statistics.mean(diffs):+11.3f} [{lo:+8.3f},{hi:+8.3f}]")
    best = min(means.values())
    chosen = min(rad for rad in radii if means[rad] <= best * 1.01)
    print(f"\nchosen radius: {chosen:g} m (lowest mean {best:.3f}; smallest radius within 1% of it)")


if __name__ == "__main__":
    main(sys.argv[1:])
