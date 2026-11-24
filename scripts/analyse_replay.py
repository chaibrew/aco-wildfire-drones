import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment.replay import load_stages, scan_anomalies, stage_report  # noqa: E402

METRICS = ["map_objective", "map_overdue_count", "true_overdue_count", "true_objective", "total_distance_m",
           "longest_route_m", "planning_s"]


def per_seed(rows, planner, drones, metric, keep=lambda r: True):
    bucket = defaultdict(list)
    for r in rows:
        if r["planner"] == planner and r["drones"] == drones and keep(r):
            bucket[r["seed"]].append(r[metric])
    return {s: statistics.fmean(v) for s, v in bucket.items()}


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    if n < 2:
        return None
    mean, sd = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    nonzero = d[np.abs(d) > 1e-12]
    p_w = 1.0 if nonzero.size == 0 else float(stats.wilcoxon(nonzero).pvalue)
    return {"n": n, "mean_a": float(np.mean([a[s] for s in seeds])), "mean_b": float(np.mean([b[s] for s in seeds])),
            "diff": float(mean), "lo": float(mean - half), "hi": float(mean + half),
            "vns_better": int((d < -1e-12).sum()), "tied": int((np.abs(d) <= 1e-12).sum()),
            "vns_worse": int((d > 1e-12).sum()), "p_wilcoxon": p_w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--stage", type=int, default=None)
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=6.0)
    a = ap.parse_args()
    rows = load_stages(Path(a.results_dir) / "stages.csv")
    drone_counts = sorted({r["drones"] for r in rows})
    seeds = sorted({r["seed"] for r in rows})
    out = [f"# Greedy vs VNS, full-mission replay ({len(seeds)} seeds, {len(rows)} stage rows)", "",
           "Difference = vns - greedy, averaged per seed; negative favours VNS.", ""]

    for drones in drone_counts:
        out += [f"## {drones} drones", "", "| metric | greedy | vns | diff (vns - greedy) | 95% CI | VNS better/tied/worse | Wilcoxon p |",
                "|---|---|---|---|---|---|---|"]
        for metric in METRICS:
            r = paired(per_seed(rows, "vns", drones, metric), per_seed(rows, "greedy", drones, metric))
            if r:
                out.append(f"| {metric} | {r['mean_b']:.3f} | {r['mean_a']:.3f} | {r['diff']:+.3f} | "
                           f"[{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['vns_better']}/{r['tied']}/{r['vns_worse']} | {r['p_wilcoxon']:.3f} |")
        whole = lambda r: r["front_on_map_fraction"] >= 0.9999
        share = sum(whole(r) for r in rows if r["drones"] == drones and r["planner"] == "greedy") / \
            max(1, sum(1 for r in rows if r["drones"] == drones and r["planner"] == "greedy"))
        r = paired(per_seed(rows, "vns", drones, "true_objective", whole),
                   per_seed(rows, "greedy", drones, "true_objective", whole))
        if r:
            out.append(f"| true_objective, whole front on map ({100 * share:.0f}% of stages) | {r['mean_b']:.3f} | {r['mean_a']:.3f} | "
                       f"{r['diff']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['vns_better']}/{r['tied']}/{r['vns_worse']} | {r['p_wilcoxon']:.3f} |")
        out.append("")

    idx = {(r["planner"], r["drones"], r["seed"], r["stage"]): r for r in rows}
    out += ["## Is true_objective planner-independent?", "",
            "| drones | stages where greedy and vns have identical true_objective | of those, share with part of the true front missing from the map | median true_objective there | median where they differ |",
            "|---|---|---|---|---|"]
    for drones in drone_counts:
        same, diff = [], []
        for (p, d, s, st), r in idx.items():
            if p == "greedy" and d == drones:
                (same if abs(r["true_objective"] - idx[("vns", d, s, st)]["true_objective"]) < 1e-9 else diff).append(r)
        out.append(f"| {drones} | {len(same)}/{len(same) + len(diff)} ({100 * len(same) / (len(same) + len(diff)):.0f}%) | "
                   f"{100 * sum(r['front_on_map_fraction'] < 0.9999 for r in same) / max(1, len(same)):.0f}% | "
                   f"{statistics.median(r['true_objective'] for r in same):.1f} | {statistics.median(r['true_objective'] for r in diff):.1f} |")
    out.append("")

    out += ["## Stages whose median jumps from the previous stage", ""]
    for metric in ("true_objective", "map_objective", "true_overdue_count", "planning_s"):
        found = scan_anomalies(rows, metric, a.threshold, top=3)
        out.append(f"**{metric}** (robust z >= {a.threshold:g}): " +
                   ("; ".join(f"{f['drones']} drones {f['planner']} stage {f['stage']} ({f['jump']:+.2f})" for f in found) or "none"))
    out.append("")

    if a.stage is not None:
        out += [f"## Stage {a.stage} against its neighbours (+-{a.window} stages, medians across seeds)", "",
                "| drones | planner | metric | at stage | neighbours | diff | seeds peaking here | snapshots delivered | points | affected fraction |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for metric in ("true_objective", "map_objective", "true_overdue_count", "planning_s"):
            for r in stage_report(rows, a.stage, a.window, metric):
                out.append(f"| {r['drones']} | {r['planner']} | {metric} | {r['median_at_stage']:.3f} | {r['median_neighbours']:.3f} | "
                           f"{r['difference']:+.3f} | {r['seeds_peaking']}/{r['seeds']} | {r['seeds_with_snapshot']}/{r['seeds']} | "
                           f"{r['median_map_points']:.0f} | {r['median_affected_fraction']:.2f} |")
        out.append("")
    text = "\n".join(out)
    (Path(a.results_dir) / "analysis.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()