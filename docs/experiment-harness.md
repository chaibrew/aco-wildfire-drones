# Baselines and experiment harness

Planners: `src/baselines/` (greedy, VNS). Controlled-change experiment: `src/experiment/`.
Scripts: `scripts/run_change_experiment.py`, `scripts/analyse_change_experiment.py`
(and `scripts/aco_experiment.py`, which now also runs greedy and VNS in the full mission replay).

## Planners

All planners are callables `router(PlanningState) -> {drone_id: [point ids]}` (see `simulation-contract.md`).

| Planner | Rule |
| --- | --- |
| `GreedyRouter` | The drone that is free first flies to the feasible point with the largest `(gap ratio on arrival) / (travel time)`. Gap ratio uses the last *planned* observation and `q_i`. One visit also reserves everything in `coverage[i]`. Ties: shorter travel, then point id. Deterministic. |
| `VNSRouter` | Start from greedy. Loop: shake the incumbent with `k` random moves (`k = 1..k_max`), descend with a randomised VND over six neighbourhoods (relocate within a route, relocate between routes, swap, 2-opt, replace a visit, insert a visit), accept if better, then `k = 1`, else `k += 1`. Stateless between replans. |

VNS shares the ACO router's objective ordering (`Metrics.score = (margin, total_overdue, distance)`) and its
evaluator. Its budget is the number of *distinct plan evaluations* (cache hits and feasibility checks are
free). The runner sets it to `ants * iterations`, the number ACO spends, so "same planning budget" holds
independent of hardware. `VNSParams(time_budget_s=...)` gives a wall-clock budget instead.
`patience` (default 100) stops the search when no new distinct plan has been found for that many iterations,
which only matters when the search space is smaller than the budget.

## Controlled-change experiment

One *paired trial* is one base scenario and one controlled change:

1. Base state `S0`: `synthetic_state(...)` (irregular loop of cells) or a real `PlanningState` captured from a
   FEDS replay (`captured_states`).
2. Warm one ACO router on `S0` for `--warmup` replans and snapshot it (pheromone, `tau0`, RNG).
3. `apply_change` turns `S0` into `S1`:
   * magnitude: share of points affected, measured with the repo's own `changes(...).fraction`;
   * locality: `concentrated` (the `m` points nearest a random centre) or `distributed` (random sample);
   * kind: `risk` (q changes), `advance` (point removed, new one a cell further out) or `mixed`.
4. Every planner replans `S1` with the same budget. The four ACO strategies are restored from the same
   snapshot and given the same RNG seed; VNS gets the same seed.
5. Each returned plan is re-scored with `fire_sim.simulation.evaluate` on `S1`, which raises on any
   violation of the horizon rule.

With `--fractions 0` all four ACO strategies produce identical plans (no change means no strategy acts);
this is a unit test.

### Run it

```sh
# pilot: choose sizes and change levels, then freeze them. Rows are labelled phase=pilot and
# ignored by the analysis unless --include-pilot is given.
python scripts/run_change_experiment.py --phase pilot --trials 3 --out outputs/pilot.csv

# final sweep (about 3 s per paired trial at 137 points; 30 trials x 10 cells ~ 15 min on one core)
python scripts/run_change_experiment.py --trials 30 --points 137 --out outputs/change_synthetic.csv

# real Camp Fire states (needs data/raw/LargeFires_2018.gpkg)
python scripts/run_change_experiment.py --source feds --trials 20 --out outputs/change_camp.csv

python scripts/analyse_change_experiment.py outputs/change_synthetic.csv
python scripts/analyse_change_experiment.py outputs/change_synthetic.csv --reference aco_retention --group locality
```

### Output columns

`<out>.csv`, one row per (trial, change, planner): the change (`target_fraction`, `realised_fraction`,
`locality`, `kind`, `dispersion`, counts of added / removed / changed points) and the result (`objective` = J,
`margin`, `total_overdue`, `overdue_count`, `mean_gap_s`, `max_gap_s`, `total_distance_m`, `longest_route_m`,
`runtime_s`, `evaluations`, `iterations`). `dispersion` is mean pairwise distance of the affected points over
that of all points (about 0.2 concentrated, about 1.0 distributed), a check that the locality factor is real.

`<out>_trace.csv`: best-so-far objective, margin and total overdue against evaluations used, per planner and
trial (the "objective trace following each change").

### Analysis

Paired by `(scenario, trial, fraction, locality)`, all metrics lower-is-better. For every planner, per
(fraction, locality) cell or pooled (`--group`): mean, median, sd, mean paired difference against `--reference`
(default `aco_full_reset`), 95% t interval, paired t and Wilcoxon p-values, and better / tied / worse counts.

## Things to know before quoting results

* **J saturates.** With a 600 s horizon, any risk-3 point (q = 300 s) that is not reached gives J >= 1 and the
  max hides everything else. Report `margin`, `total_overdue` and `overdue_count` next to J. The ACO router
  already ranks plans with `margin`, then `total_overdue`.
* **VNS at equal budget.** VNS gets 1,200 evaluations, the same as ACO. On tiny instances (4 points,
  2 drones, 1,200 evaluations) it matches the exhaustive optimum on `margin` in 15 of 15 instances. On
  `total_overdue` it matches in 10 of 15 and is 2–17 % worse in the other 5.
* **The change is instantaneous.** `S1` has the same time and drone positions as `S0`; only the points differ.
  This isolates the pheromone question, but it is not the same as the mission replay, where drones move and
  changes arrive with 12-hour snapshots. `scripts/aco_experiment.py` is the replay.
* **Multiple comparisons.** The analysis gives raw and Holm-adjusted p-values. The report uses the Holm values.
* **`runtime_s` with `--workers > 1`** is only comparable within one run.
