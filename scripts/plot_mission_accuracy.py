import argparse
import csv
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [("perimeter_error_m", "perimeter error (m)"), ("growth_found_fraction", "growth since snapshot found"),
          ("true_objective", "true-front J")]
STYLE = {"greedy": ("#8a8a8a", "--"), "vns": ("#555555", "-"), "smoke_router": ("#c0c0c0", ":"),
         "aco_retention": ("#2a6fbb", "-"), "aco_full_reset": ("#d6402b", "-"),
         "aco_fast_evaporation": ("#e08a1e", "-"), "aco_selective_reset": ("#2f9e44", "-")}


def main():
    p = argparse.ArgumentParser(description="Plot map accuracy and true J over a mission replay.")
    p.add_argument("stages_csv")
    p.add_argument("--planners", nargs="+", default=None)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    values = defaultdict(lambda: defaultdict(list))
    snapshots = set()
    with open(a.stages_csv) as fh:
        for r in csv.DictReader(fh):
            if a.planners and r["planner"] not in a.planners:
                continue
            t = float(r["time_s"]) / 3600
            if r["snapshot_delivered"] and int(r["stage"]) > 0:
                snapshots.add(t)
            for key, _ in PANELS:
                if r[key] not in ("", "nan"):
                    values[(r["planner"], key)][t].append(float(r[key]))
    planners = sorted({pl for pl, _ in values}, key=list(STYLE).index)
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(9, 2.6 * len(PANELS)), sharex=True)
    for ax, (key, label) in zip(axes, PANELS):
        for pl in planners:
            series = values[(pl, key)]
            ts = sorted(series)
            colour, ls = STYLE.get(pl, ("black", "-"))
            ax.plot(ts, [statistics.mean(series[t]) for t in ts], color=colour, ls=ls, lw=1.4, label=pl)
        for s in snapshots:
            ax.axvline(s, color="black", lw=0.6, alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("hours into the mission")
    axes[0].legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(a.out)


if __name__ == "__main__":
    main()
