from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tuning import SEEDS, UNTUNED_PARAMS, camp_states, load_params, make_instances  # noqa: E402
from tuning.run import ParallelEvaluator, confirmation_summary, result_rows, setting_means, write_csv  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("settings", nargs="+", help="NAME=defaults or NAME=path/to/params.json; first = reference")
    p.add_argument("--out", required=True, help="raw results csv; _summary.csv and _means.csv are written beside it")
    p.add_argument("--camp", action="store_true")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--ants", type=int, default=20)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--drones", type=int, default=6)
    p.add_argument("--gpkg", default="data/raw/LargeFires_2018.gpkg")
    p.add_argument("--cache", default="data/cache")
    return p.parse_args()


def main():
    a = parse_args()
    settings = []
    for item in a.settings:
        name, _, spec = item.partition("=")
        params = UNTUNED_PARAMS if spec in ("", "defaults") else load_params(spec)
        settings.append((name, params.with_(n_ants=a.ants, max_iterations=a.iterations)))
    seeds = SEEDS["confirm"]
    camp = camp_states(seeds, gpkg=a.gpkg, cache=a.cache, drones=a.drones) if a.camp else None
    instances = make_instances(seeds, a.instances, camp, drones=a.drones)
    rows = []
    with ParallelEvaluator(instances, a.workers, warmup=a.warmup) as ev:
        for name, params in settings:
            for i, res in enumerate(ev.across_instances(params)):
                rows += result_rows(instances, i, name, res)
            print(f"{name} done", flush=True)
    stem = a.out[:-4] if a.out.endswith(".csv") else a.out
    write_csv(a.out, rows)
    ref, others = settings[0][0], [n for n, _ in settings[1:]]
    summary = confirmation_summary(rows, ref, others)
    write_csv(f"{stem}_summary.csv", summary)
    means = setting_means(rows)
    write_csv(f"{stem}_means.csv", means)

    print(f"\n{'setting':>16s} {'margin':>7s} {'tot.over':>8s} {'n.over':>7s}")
    for m in means:
        print(f"{m['setting']:>16s} {m['margin']:7.3f} {m['total_overdue']:8.2f} {m['overdue_count']:7.2f}")
    print(f"\nvs {ref} (better/tied/worse, mean change, Wilcoxon p):")
    for r in summary:
        if r["measure"] != "rank_key" and (r["strategy"] == "all" or r["measure"] == "total_overdue"):
            print(f"  {r['setting']:>12s} {r['strategy']:>17s} {r['measure']:>14s} "
                  f"{r['better']:3d}/{r['tied']:3d}/{r['worse']:3d}  {r['mean_change']:+9.3f}  p={r['p']:.2g}")
    print(f"\nwrote {a.out}, {stem}_summary.csv, {stem}_means.csv")


if __name__ == "__main__":
    main()
