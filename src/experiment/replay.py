"""Full-mission replay of the baseline planners, with every replanning stage saved.

`scripts/aco_experiment.py` keeps only one summary row per mission, so a single odd stage (for
example "stage 40") cannot be inspected afterwards.  This module runs the same missions but
writes every stage to stages.csv, tagged with planner, drone count and seed, plus one summary row
per mission to summary.csv.  It then provides tools to look at a stage in detail
(`stage_report`) and to scan for stages that jump away from their neighbours (`scan_anomalies`).

Design notes
------------
* The seed fixes the ground-truth fire noise (and VNS's random generator), and the same seed gives
  every planner and every drone count the same fire, so results are paired by seed.
* Greedy is deterministic.  VNS gets `vns_evaluations` plan evaluations per replan; the default in
  the scripts is ants * iterations, the same budget unit the ACO router uses.
* Missions are written one at a time and `resume=True` skips missions already in summary.csv.
  Stage rows of a mission that crashed before its summary row was written are dropped on resume.
"""
from __future__ import annotations

import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

from aco_router import summary
from baselines import GreedyRouter, VNSParams, VNSRouter
from fire_sim.mission import MissionConfig, run_mission

TEXT_COLUMNS = {"planner", "snapshot_delivered"}
INT_COLUMNS = {"drones", "seed", "stage", "block"}


def baseline_planners(vns_evaluations: int) -> dict:
    """{name: factory(seed) -> router} for the two baselines."""
    return {"greedy": lambda seed: GreedyRouter(),
            "vns": lambda seed: VNSRouter(VNSParams(max_evaluations=vns_evaluations), seed=seed)}


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def _append(path: Path, rows: list[dict]):
    if not rows:
        return
    new = not path.exists() or path.stat().st_size == 0
    fields = list(rows[0]) if new else next(csv.reader(path.open(newline="")))
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, restval="")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def _done(summary_csv: Path) -> set:
    if not summary_csv.exists():
        return set()
    with summary_csv.open(newline="") as stream:
        return {(r["planner"], int(r["drones"]), int(r["seed"])) for r in csv.DictReader(stream)}


def _drop_incomplete(stages_csv: Path, keep: set):
    """Remove stage rows whose mission has no summary row (a run that stopped part-way)."""
    if not stages_csv.exists():
        return 0
    with stages_csv.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    good = [r for r in rows if (r["planner"], int(r["drones"]), int(r["seed"])) in keep]
    if len(good) != len(rows):
        with stages_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(good)
    return len(rows) - len(good)


def run_replay(field_for_seed, planners: dict, drone_counts, seeds, hours: float, out_dir, *,
               cell_size_m: float = 1000.0, horizon_s: float = 600, resume: bool = False, log=print):
    """Fly each planner at each drone count on each seed's fire for `hours` hours."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stages_csv, summary_csv = out / "stages.csv", out / "summary.csv"
    if resume:
        done = _done(summary_csv)
        dropped = _drop_incomplete(stages_csv, done)
        if dropped:
            log(f"resume: dropped {dropped} stage rows from unfinished missions")
    else:
        done = set()
        for path in (stages_csv, summary_csv):
            path.unlink(missing_ok=True)
    total = len(seeds) * len(drone_counts) * len(planners)
    count = len(done)
    for seed in seeds:
        todo = [(d, n) for d in drone_counts for n in planners if (n, d, seed) not in done]
        if not todo:
            continue
        field = field_for_seed(seed)                      # one fire per seed, shared by every run
        for drones, name in todo:
            config = MissionConfig(cell_size_m=cell_size_m, drones=drones, horizon_s=horizon_s)
            started = time.time()
            rows, _ = run_mission(field, planners[name](seed), config, stop_s=field.start_s + hours * 3600)
            wall = time.time() - started
            tag = {"planner": name, "drones": drones, "seed": seed}
            _append(stages_csv, [{**tag, **r} for r in rows])
            _append(summary_csv, [{**tag, **summary(rows), "stages": len(rows), "wall_s": round(wall, 2)}])
            count += 1
            log(f"[{count}/{total}] seed {seed} drones {drones:2d} {name:8s} {len(rows)} stages in {wall:.1f}s")
    return {"stages": stages_csv, "summary": summary_csv}


# --------------------------------------------------------------------------- #
# Looking at stages
# --------------------------------------------------------------------------- #
def load_stages(path) -> list[dict]:
    rows = []
    with Path(path).open(newline="") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in TEXT_COLUMNS:
                    row[key] = value
                elif key in INT_COLUMNS:
                    row[key] = int(float(value))
                else:
                    try:
                        row[key] = float(value) if value != "" else float("nan")
                    except ValueError:
                        row[key] = value
            rows.append(row)
    return rows


def _series(rows, drones, planner, metric):
    """{stage: [values across seeds]} for one (drones, planner)."""
    out = defaultdict(list)
    for r in rows:
        if r["drones"] == drones and r["planner"] == planner:
            out[r["stage"]].append(r[metric])
    return out


def stage_report(rows, stage: int, window: int = 3, metric: str = "true_objective") -> list[dict]:
    """For every (drones, planner), compare `stage` with the stages around it.

    Reported per run group: the median `metric` at the stage, the median over the neighbouring
    stages (stage-window .. stage+window, excluding the stage itself), their difference, how many
    seeds peak at this stage, the seed with the largest value, and how many seeds received a new
    satellite snapshot at this stage.  If all seeds peak here and a snapshot arrives, the anomaly
    is systematic (tied to the data); if one seed dominates, it is a single-run outlier.
    """
    out = []
    for drones, planner in sorted({(r["drones"], r["planner"]) for r in rows}):
        series = _series(rows, drones, planner, metric)
        if stage not in series:
            continue
        neighbours = [v for s in range(stage - window, stage + window + 1) if s != stage and s in series
                      for v in series[s]]
        per_seed = {r["seed"]: r for r in rows
                    if r["drones"] == drones and r["planner"] == planner and r["stage"] == stage}
        peaks = 0
        for seed in per_seed:
            others = [r[metric] for r in rows if r["drones"] == drones and r["planner"] == planner
                      and r["seed"] == seed and 0 < abs(r["stage"] - stage) <= window]
            peaks += bool(others) and per_seed[seed][metric] > max(others)   # strictly higher than all neighbours
        top = max(per_seed.items(), key=lambda kv: kv[1][metric])
        at = statistics.median(series[stage])
        around = statistics.median(neighbours) if neighbours else float("nan")
        total = sum(series[stage])
        out.append({"drones": drones, "planner": planner, "stage": stage, "metric": metric,
                    "median_at_stage": at, "median_neighbours": around, "difference": at - around,
                    "seeds": len(per_seed), "seeds_peaking": peaks, "top_seed": top[0],
                    "top_seed_value": top[1][metric],
                    "top_seed_share_of_sum": top[1][metric] / total if total else float("nan"),
                    "seeds_with_snapshot": sum(bool(str(r["snapshot_delivered"]).strip()) for r in per_seed.values()),
                    "median_map_points": statistics.median(r["map_points"] for r in per_seed.values()),
                    "median_affected_fraction": statistics.median(r["affected_union_fraction"]
                                                                  for r in per_seed.values())})
    return out


def scan_anomalies(rows, metric: str = "true_objective", threshold: float = 4.0, top: int = 5) -> list[dict]:
    """Find stages whose median `metric` jumps unusually far from the previous stage.

    For each (drones, planner) the series of per-stage medians across seeds is differenced and a
    robust z-score  (diff - median(diffs)) / (1.4826 * MAD(diffs))  is computed.  Stages with
    |z| >= threshold are returned, largest first.  Constant series are skipped.
    """
    found = []
    for drones, planner in sorted({(r["drones"], r["planner"]) for r in rows}):
        series = _series(rows, drones, planner, metric)
        stages = sorted(series)
        med = [statistics.median(series[s]) for s in stages]
        diffs = [b - a for a, b in zip(med, med[1:])]
        if len(diffs) < 3:
            continue
        centre = statistics.median(diffs)
        mad = statistics.median(abs(d - centre) for d in diffs) * 1.4826
        if mad == 0:
            mad = 1e-9 if any(d != centre for d in diffs) else 0
        if mad == 0:
            continue
        flagged = sorted(((abs((d - centre) / mad), stages[i + 1], d) for i, d in enumerate(diffs)
                          if abs((d - centre) / mad) >= threshold), reverse=True)[:top]
        found += [{"drones": drones, "planner": planner, "stage": s, "jump": d, "robust_z": z,
                   "metric": metric} for z, s, d in flagged]
    return found