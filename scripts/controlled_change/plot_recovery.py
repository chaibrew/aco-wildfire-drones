import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiment.stats import load_trace, recovery_summary  # noqa: E402

COLOURS = {"aco_retention": "#2a6fbb", "aco_fast_evaporation": "#e08a1e", "aco_selective_reset": "#2f9e44",
           "vns": "#8a8a8a"}
LABELS = {"aco_retention": "Retention", "aco_fast_evaporation": "Fast evaporation",
          "aco_selective_reset": "Selective reset", "vns": "VNS"}


def main():
    p = argparse.ArgumentParser(description="Plot each planner's best plan minus full reset's against the evaluations after a change.")
    p.add_argument("csv", nargs="+")
    p.add_argument("--titles", nargs="+")
    p.add_argument("--value", default="best_total_overdue")
    p.add_argument("--planners", nargs="+", default=list(COLOURS), choices=list(COLOURS))
    p.add_argument("--out", required=True)
    a = p.parse_args()
    checkpoints = tuple(range(20, 1201, 20))
    fig, axes = plt.subplots(1, len(a.csv), figsize=(5.2 * len(a.csv), 3.8), sharey=True, squeeze=False)
    for k, (ax, path) in enumerate(zip(axes[0], a.csv)):
        trace = load_trace(path[:-4] + "_trace.csv")
        rows = recovery_summary(trace, a.value, "aco_full_reset", checkpoints)
        for planner, colour in ((pl, COLOURS[pl]) for pl in a.planners):
            pts = [r for r in rows if r["planner"] == planner and "mean_diff" in r]
            if not pts:
                continue
            x = [r["evaluations"] for r in pts]
            ax.plot(x, [r["mean_diff"] for r in pts], color=colour, label=LABELS[planner], lw=1.8)
            ax.fill_between(x, [r["ci_lo"] for r in pts], [r["ci_hi"] for r in pts], color=colour, alpha=0.15, lw=0)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("plan evaluations after the change")
        ax.set_title(a.titles[k] if a.titles and k < len(a.titles) else os.path.basename(path), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(f"{a.value.replace('best_', '').replace('_', ' ')} minus full reset")
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(a.out)


if __name__ == "__main__":
    main()
