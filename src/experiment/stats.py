"""Paired analysis of a results CSV written by the controlled-change runner.

Pairing: rows with the same (scenario, trial, target_fraction, locality) share the base
scenario, the change, the pre-change pheromone and the RNG seed, so each planner is
compared with the reference on the *same* trial. All metrics are "lower is better".

Independent unit.  One trial is one base scenario (`synthetic_state(seed)`, or one captured FEDS
state with its own change draw).  The SAME trial is reused for every target fraction and both
localities, so rows that share (scenario, trial) are NOT independent.  Whenever a grouping pools
several rows per trial (`locality`, `fraction`, `all`), the paired differences are first averaged
within each (scenario, trial) cluster and the confidence interval and p-values use one number per
cluster.  Note `scenario` is only a label (e.g. "synthetic") so the cluster key is the pair
(scenario, trial); using `scenario` alone would collapse a whole run into a single cluster.

Sign convention for differences and interactions: planner minus reference, lower is better, so a
NEGATIVE value means the planner is better.
"""
from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict

try:  # scipy is a dependency of fire_sim; guard anyway so the module imports without it
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None

METRICS = ("margin", "total_overdue", "overdue_count", "objective", "max_gap_s", "mean_gap_s",
           "total_distance_m", "longest_route_m", "runtime_s")
GROUPINGS = {
    "cell": ("target_fraction", "locality"),
    "locality": ("locality",),
    "fraction": ("target_fraction",),
    "all": (),
}
PLANNER_ORDER = ("greedy", "vns", "aco_retention", "aco_full_reset", "aco_fast_evaporation", "aco_selective_reset")


def load_rows(path: str, include_pilot: bool = False) -> list[dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not include_pilot:
        rows = [r for r in rows if r.get("phase", "final") != "pilot"]
    for r in rows:
        for k, v in r.items():
            if k in METRICS or k in ("target_fraction", "realised_fraction", "dispersion", "evaluations"):
                r[k] = float(v)
    return rows


def _t_ci(diffs: list[float]) -> tuple[float, float]:
    n = len(diffs)
    if n < 2:
        return (float("nan"), float("nan"))
    m = statistics.mean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n)
    crit = _sps.t.ppf(0.975, n - 1) if _sps else 1.96
    return (m - crit * se, m + crit * se)


def _p_values(diffs: list[float]) -> tuple[float, float]:
    if _sps is None or len(diffs) < 2 or all(d == 0 for d in diffs):
        return (float("nan"), float("nan"))
    t_p = float(_sps.ttest_1samp(diffs, 0.0).pvalue) if statistics.pstdev(diffs) > 0 else float("nan")
    try:
        w_p = float(_sps.wilcoxon(diffs).pvalue)
    except ValueError:
        w_p = float("nan")
    return (t_p, w_p)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the original order. NaN stays NaN and is not counted."""
    adjusted = [float("nan")] * len(p_values)
    valid = sorted(((i, p) for i, p in enumerate(p_values) if math.isfinite(p)), key=lambda t: t[1])
    m, running = len(valid), 0.0
    for rank, (i, p) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = running
    return adjusted


def paired_summary(rows: list[dict], metric: str, reference: str, grouping: str = "cell") -> list[dict]:
    keys = GROUPINGS[grouping]
    cell_rows = defaultdict(lambda: defaultdict(dict))  # group -> planner -> pair id -> value
    for r in rows:
        group = tuple(r[k] for k in keys)
        pair = (r["scenario"], r["trial"], r["target_fraction"], r["locality"])
        cell_rows[group][r["planner"]][pair] = r[metric]
    out = []
    for group in sorted(cell_rows):
        by_planner = cell_rows[group]
        ref = by_planner.get(reference, {})
        for planner in PLANNER_ORDER:
            vals = by_planner.get(planner)
            if not vals:
                continue
            xs = list(vals.values())
            row = {**dict(zip(keys, group)), "planner": planner, "n": len(xs), "mean": statistics.mean(xs),
                   "median": statistics.median(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0}
            if planner != reference and ref:
                pairs = [p for p in vals if p in ref]
                # pair = (scenario, trial, target_fraction, locality). Average the paired differences within
                # each independent (scenario, trial) cluster before doing any inference.
                by_cluster = defaultdict(list)
                for p in pairs:
                    by_cluster[(p[0], p[1])].append(vals[p] - ref[p])
                diffs = [statistics.mean(v) for v in by_cluster.values()]
                lo, hi = _t_ci(diffs)
                t_p, w_p = _p_values(diffs)
                row.update(n_pairs=len(pairs), n_clusters=len(diffs),
                           mean_diff=statistics.mean(diffs) if diffs else float("nan"),
                           ci_lo=lo, ci_hi=hi, t_p=t_p, wilcoxon_p=w_p,
                           better=sum(d < 0 for d in diffs), tied=sum(d == 0 for d in diffs),
                           worse=sum(d > 0 for d in diffs))
            out.append(row)
    # Holm correction across the planners compared with the reference inside each group.
    for group in {tuple(r.get(k) for k in keys) for r in out}:
        members = [r for r in out if tuple(r.get(k) for k in keys) == group and "t_p" in r]
        for field, adjusted in (("t_p_holm", holm_adjust([r["t_p"] for r in members])),
                                ("wilcoxon_p_holm", holm_adjust([r["wilcoxon_p"] for r in members]))):
            for r, value in zip(members, adjusted):
                r[field] = value
    return out


def format_summary(summary: list[dict], metric: str, reference: str, grouping: str) -> str:
    keys = GROUPINGS[grouping]
    lines = [f"{metric} (lower is better); differences are planner - {reference}, averaged within each trial "
             f"(n = independent trials); t p(H) is Holm-adjusted across planners", ""]
    head = "  ".join(f"{k:>16s}" for k in keys)
    lines.append(f"{head}  {'planner':22s} {'mean':>9s} {'median':>9s} {'sd':>8s} {'diff':>9s} "
                 f"{'95% CI':>19s} {'t p':>7s} {'t p(H)':>7s} {'W p':>7s} {'b/t/w':>9s} {'n':>4s}")
    for r in summary:
        lead = "  ".join(f"{r[k]:>16}" if not isinstance(r[k], float) else f"{r[k]:>16.2f}" for k in keys)
        tail = ""
        if "mean_diff" in r:
            tail = (f" {r['mean_diff']:+9.3f} [{r['ci_lo']:+8.3f},{r['ci_hi']:+8.3f}] {r['t_p']:7.3f} "
                    f"{r.get('t_p_holm', float('nan')):7.3f} {r['wilcoxon_p']:7.3f} "
                    f"{r['better']:>3d}/{r['tied']}/{r['worse']} {r['n_clusters']:>4d}")
        lines.append(f"{lead}  {r['planner']:22s} {r['mean']:9.3f} {r['median']:9.3f} {r['sd']:8.3f}{tail}")
    return "\n".join(lines)


def trace_summary(trace_rows: list[dict], checkpoints=(1, 10, 30, 60, 100, 150), value="best_margin") -> list[dict]:
    """Median best-so-far value at fixed evaluation counts: the 'objective trace after a change'."""
    runs = defaultdict(list)  # (planner, trial, fraction, locality) -> [(evals, value)]
    for r in trace_rows:
        if r.get("phase", "final") == "pilot":
            continue
        runs[(r["planner"], r["scenario"], r["trial"], float(r["target_fraction"]), r["locality"])].append(
            (float(r["evaluations"]), float(r[value])))
    per_planner = defaultdict(lambda: defaultdict(list))
    for (planner, *_), pts in runs.items():
        pts.sort()
        for cp in checkpoints:
            seen = [v for e, v in pts if e <= cp]
            if seen:
                per_planner[planner][cp].append(min(seen))
    return [{"planner": p, "evaluations": cp, f"median_{value}": statistics.median(vs), "n": len(vs)}
            for p in PLANNER_ORDER if p in per_planner for cp, vs in sorted(per_planner[p].items())]


def load_trace(path: str) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def change_interaction(rows: list[dict], metric: str, planner: str = "aco_selective_reset",
                       reference: str = "aco_full_reset", factor: str = "locality",
                       fraction_split: float = 0.5) -> dict:
    """Does `planner` do relatively better than `reference` under one kind of change than another?

    This tests the project's research question directly.  For every (trial, fraction, locality) the
    difference d = planner - reference is taken.  Within each independent (scenario, trial) cluster
    the d values are averaged on each side of the factor:
        factor="locality": concentrated (a) versus distributed (b)
        factor="fraction": fraction <= fraction_split (a, small) versus > fraction_split (b, large)
    and interaction = mean_d(a) - mean_d(b).  The distribution of per-cluster interactions is then
    tested against zero.  NEGATIVE interaction: the planner's difference to the reference is lower
    (better) under a than under b, e.g. selective reset gains more after concentrated/small changes.
    Clusters that lack either side are skipped.  `fraction_split` defaults to 0.5, the same cut-off
    the ACO router uses to call a change LOCAL (aco_router.config.LOCAL_CHANGE_MAX_FRACTION).
    """
    if factor not in ("locality", "fraction"):
        raise ValueError("factor must be 'locality' or 'fraction'")
    values = defaultdict(dict)
    for r in rows:
        if r["planner"] in (planner, reference):
            values[(r["scenario"], r["trial"], r["target_fraction"], r["locality"])][r["planner"]] = r[metric]
    sides = defaultdict(lambda: {"a": [], "b": []})
    for (scenario, trial, fraction, locality), v in values.items():
        if planner not in v or reference not in v:
            continue
        if factor == "locality":
            side = "a" if locality == "concentrated" else "b"
        else:
            side = "a" if fraction <= fraction_split else "b"
        sides[(scenario, trial)][side].append(v[planner] - v[reference])
    da, db, inter = [], [], []
    for c in sides.values():
        if c["a"] and c["b"]:
            da.append(statistics.mean(c["a"]))
            db.append(statistics.mean(c["b"]))
            inter.append(da[-1] - db[-1])
    labels = (("concentrated", "distributed") if factor == "locality"
              else (f"fraction<={fraction_split:g}", f"fraction>{fraction_split:g}"))
    lo, hi = _t_ci(inter)
    t_p, w_p = _p_values(inter)
    if len(inter) < 3:
        reading = "too few trials to say anything"
    elif hi < 0:
        reading = f"{planner} does relatively better under {labels[0]} (CI excludes 0)"
    elif lo > 0:
        reading = f"{planner} does relatively better under {labels[1]} (CI excludes 0)"
    else:
        reading = "no clear difference between the two conditions (CI includes 0)"
    return {"factor": factor, "metric": metric, "planner": planner, "reference": reference,
            "label_a": labels[0], "label_b": labels[1], "n_clusters": len(inter),
            "mean_diff_a": statistics.mean(da) if da else float("nan"),
            "mean_diff_b": statistics.mean(db) if db else float("nan"),
            "mean_interaction": statistics.mean(inter) if inter else float("nan"),
            "ci_lo": lo, "ci_hi": hi, "t_p": t_p, "wilcoxon_p": w_p,
            "trials_favouring_a": sum(i < 0 for i in inter), "reading": reading}


def format_interactions(results: list[dict]) -> str:
    lines = ["interaction = (planner - reference under A) - (planner - reference under B); negative = planner "
             "relatively better under A", ""]
    for r in results:
        lines.append(f"{r['metric']:>14s}  {r['planner']} vs {r['reference']}  A={r['label_a']}  B={r['label_b']}")
        lines.append(f"{'':>16s}diff A {r['mean_diff_a']:+.3f}  diff B {r['mean_diff_b']:+.3f}  interaction "
                     f"{r['mean_interaction']:+.3f}  95% CI [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}]  t p {r['t_p']:.3f}  "
                     f"W p {r['wilcoxon_p']:.3f}  trials favouring A {r['trials_favouring_a']}/{r['n_clusters']}")
        lines.append(f"{'':>16s}-> {r['reading']}")
    return "\n".join(lines)

def _incumbent_at(points: list[tuple[float, float]], checkpoint: float) -> float | None:
    value = None
    for e, v in points:
        if e > checkpoint:
            break
        value = v
    return value


def recovery_summary(trace_rows: list[dict], value: str = "best_total_overdue", reference: str = "aco_full_reset",
                     checkpoints=(20, 60, 120, 300, 600, 1200), include_pilot: bool = False) -> list[dict]:
    runs = defaultdict(list)
    for r in trace_rows:
        if not include_pilot and r.get("phase", "final") == "pilot":
            continue
        key = (r["planner"], r["scenario"], r["trial"], float(r["target_fraction"]), r["locality"])
        runs[key].append((float(r["evaluations"]), float(r[value])))
    at = defaultdict(dict)
    for (planner, *pair), pts in runs.items():
        pts.sort()
        for cp in checkpoints:
            v = _incumbent_at(pts, cp)
            if v is not None:
                at[(planner, cp)][tuple(pair)] = v
    out = []
    for cp in checkpoints:
        ref = at.get((reference, cp), {})
        members = []
        for planner in PLANNER_ORDER:
            vals = at.get((planner, cp))
            if not vals:
                continue
            row = {"evaluations": cp, "planner": planner, "n": len(vals), "median": statistics.median(vals.values()),
                   "mean": statistics.mean(vals.values())}
            if planner != reference and ref:
                by_cluster = defaultdict(list)
                for pair, v in vals.items():
                    if pair in ref:
                        by_cluster[(pair[0], pair[1])].append(v - ref[pair])
                diffs = [statistics.mean(d) for d in by_cluster.values()]
                lo, hi = _t_ci(diffs)
                t_p, w_p = _p_values(diffs)
                row.update(n_clusters=len(diffs), mean_diff=statistics.mean(diffs) if diffs else float("nan"),
                           ci_lo=lo, ci_hi=hi, t_p=t_p, wilcoxon_p=w_p, better=sum(d < 0 for d in diffs),
                           worse=sum(d > 0 for d in diffs))
                members.append(row)
            out.append(row)
        for r, adj in zip(members, holm_adjust([r["t_p"] for r in members])):
            r["t_p_holm"] = adj
    return out


def format_recovery(summary: list[dict], value: str, reference: str) -> str:
    lines = [f"recovery: {value} of the best plan after k evaluations (lower is better); diff = planner - "
             f"{reference}, averaged within each trial; t p(H) Holm-adjusted across planners at each k", "",
             f"{'evals':>6s}  {'planner':22s} {'median':>9s} {'diff':>9s} {'95% CI':>19s} {'t p(H)':>7s} {'b/w':>7s}"]
    for r in summary:
        tail = ""
        if "mean_diff" in r:
            tail = (f" {r['mean_diff']:+9.3f} [{r['ci_lo']:+8.3f},{r['ci_hi']:+8.3f}] {r.get('t_p_holm', float('nan')):7.3f}"
                    f" {r['better']:>3d}/{r['worse']:<3d}")
        lines.append(f"{r['evaluations']:>6g}  {r['planner']:22s} {r['median']:9.3f}{tail}")
    return "\n".join(lines)


def reset_share_summary(rows: list[dict], metric: str = "total_overdue",
                        bins=(0.0, 0.1, 0.25, 0.5, 1.01)) -> list[dict]:
    by_pair = defaultdict(dict)
    share = {}
    for r in rows:
        pair = (r["scenario"], r["trial"], r["target_fraction"], r["locality"])
        by_pair[pair][r["planner"]] = r[metric]
        if r["planner"] == "aco_selective_reset" and r.get("reset_points") not in ("", None):
            share[pair] = float(r["reset_points"]) / max(1.0, float(r["n_after"]))
    out = []
    for lo, hi in zip(bins, bins[1:]):
        for ref in ("aco_retention", "aco_full_reset"):
            by_cluster = defaultdict(list)
            for pair, s in share.items():
                v = by_pair[pair]
                if lo <= s < hi and "aco_selective_reset" in v and ref in v:
                    by_cluster[(pair[0], pair[1])].append(v["aco_selective_reset"] - v[ref])
            diffs = [statistics.mean(d) for d in by_cluster.values()]
            if not diffs:
                continue
            ci_lo, ci_hi = _t_ci(diffs)
            out.append({"share_lo": lo, "share_hi": min(hi, 1.0), "reference": ref, "n_clusters": len(diffs),
                        "n_pairs": sum(len(d) for d in by_cluster.values()), "mean_diff": statistics.mean(diffs),
                        "ci_lo": ci_lo, "ci_hi": ci_hi})
    return out
