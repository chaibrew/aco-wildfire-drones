# ACO parameter tuning

This document describes how the ACO router's parameters were tuned, the results of each step, and the setting that was chosen. The code is in `src/tuning/racing.py` (F-Race and the iterated race, independent of the ACO), `src/tuning/tuner.py` (the parameter space, tuning instances, scoring and result tables) and `scripts/tune_aco.py` (one subcommand per step). The raw results are written to `outputs/tuning/` (not in git; the commands at the end regenerate them, and every step is deterministic, so a rerun gives the same numbers).

## 1. Aims

Before tuning, only β, λ and the use of MAX–MIN bounds had been chosen deliberately, on small brute-force test problems. The other parameters were common defaults, and the experiments used a small planning budget (10 ants × 15 iterations per replan). Tuning had four aims:

1. Fix the planning budget (ants × iterations) at the point where more computation stops improving the plans.
2. Choose one setting of the shared parameters (α, β, λ, ρ, τ_min ratio and ε) that works well for all four pheromone strategies. One shared setting keeps the strategy comparison fair, because no strategy gets parameters tuned for it alone.
3. Choose fast evaporation's own parameters (ρ_high and k) with the shared parameters fixed.
4. Confirm the chosen setting against the old defaults on fresh instances, so that the improvement is measured on data that was not used to choose it.

The symbols are those of the router (`src/aco_router/config.py`, `ACOParams`):

| Symbol | Field | Meaning |
|---|---|---|
| α | `alpha` | exponent on the pheromone τ_ij in the move probability |
| β | `beta` | exponent on the heuristic η_ij = 1/d_ij (preference for nearby points) |
| λ | `urgency_weight` | exponent on the urgency term (ε + u_j), where u_j is how long point j will have gone unobserved when the drone arrives, as a fraction of its revisit period q_j |
| ε | `urgency_floor` | small constant added to u_j so that a just-observed point can still be chosen |
| ρ | `rho` | evaporation rate: the fraction of pheromone removed each iteration |
| τ_min ratio | `tau_min_ratio` | lower pheromone bound as a fraction of the upper bound, τ_min = ratio × τ_max |
| ρ_high, k | `rho_high`, `fast_evaporation_iterations` | fast evaporation only: after a change, ρ is raised to ρ_high for k iterations |

## 2. Separate seeds (pilot)

Each tuning step uses its own instance seeds and its own captured Camp Fire states. None of them overlaps the reported experiments, which use seeds from 0 and the Camp Fire replay with fire seed 0 (stages 6, 40 and 80). This means that no run used to choose a parameter is reused to report a result. `make_instances` refuses seeds below 1000, and a test checks that the seed blocks are disjoint.

| Step | Instance seeds | Camp Fire replay (fire seed, stages) |
|---|---|---|
| Budget check | 2000–2019 | 2000; 20, 60, 100 |
| Shared race (F-Race) | 1000–1029 | 1000; 20, 60, 100 |
| Iterated race (irace-style) | 1200–1229 | 1200; 25, 65, 105 |
| Fast-evaporation race | 1500–1529 | 1500; 20, 60, 100 |
| Confirmation | 2100–2119 | 2100; 30, 70, 110 |

## 3. How a setting is scored

**A tuning instance** is one controlled change, built with the experiment harness (`src/experiment`):

1. Take a base state. Even-numbered instances use a synthetic loop of 137 points (137 is the median number of monitoring points in the Camp Fire replay). Odd-numbered instances use the captured Camp Fire states in turn. All states have 6 drones.
2. Warm up one ACO router (retention strategy) on the base state for 5 replans, and save a snapshot of its pheromone.
3. Apply one change with `experiment.perturb.apply_change`. The change's size (fraction of points affected, from {0.05, 0.10, 0.20, 0.40, 0.60}) and locality (concentrated or distributed) are drawn per instance from the instance seed.
4. For each of the four strategies, restore the snapshot, reseed the random number generator with the same seed, and replan once on the changed state.
5. Score each plan with the simulator's own evaluator (`experiment.runner.score_routes`), which gives the margin, the total overdue and the overdue count.

The three measures are defined from P_i, the longest gap between observations of point i over the plan, and q_i, its revisit period:

- **margin** = max_i P_i/q_i, the worst gap relative to its revisit period;
- **total overdue** = Σ_i max(0, P_i/q_i − 1), the summed amount by which points exceed their revisit periods;
- **overdue count** = the number of points with P_i > q_i.

**Ranking.** Plans are ordered the way the router orders them: by margin first, and by total overdue when the margins are equal. This uses the single sort key `round(margin, 6) × 10^6 + total overdue`. The router's other cost, margin + mean overdue, was not used, because in these states the margin is set mostly by points that are already overdue before the stage starts, so that cost hardly changes between settings and the race cannot separate them.

**Blocks.** In the race, each instance is one block. Within a block, the candidate settings are ranked separately for each strategy, and the four ranks are averaged; that average is the candidate's score for the block. The four strategies are not treated as separate blocks, because they share one warm-up and one change and are therefore not independent. (Treating them as separate blocks made the Friedman test act as if it had four times as much evidence as it had.)

## 4. Racing methods

**F-Race** (Birattari et al., 2002) evaluates all candidate settings on one instance at a time. From the fifth instance on (`first_test` = 5), after each new instance a Friedman test checks whether the candidates' mean ranks differ by more than chance would explain. The Friedman test is a non-parametric test for several methods measured on the same set of problems. If it rejects at α = 0.05, a Conover post-hoc test drops every candidate whose rank sum is worse than the best rank sum by more than a critical difference. When only two candidates remain, a Wilcoxon signed-rank test is used instead. These are the statistics used by irace (López-Ibáñez et al., 2016). After a candidate is dropped, the earlier blocks are re-ranked without it. The race stops when one candidate is left or the instances run out; the survivor with the best mean rank is the race's pick.

**Iterated race.** irace repeats the race. After each race it keeps the best survivors ("elites"), samples new candidates near them, and narrows the sampling spread, so later races search a smaller region around the settings that did well. The irace package runs in R, so a small Python version was written (`IteratedRace` in `src/tuning/racing.py`). It follows irace's default rules: 2 + ⌊log₂ 6⌋ = 4 iterations for 6 parameters; the remaining budget is split evenly over the remaining iterations; up to 4 elites are kept; a new candidate takes an elite as its parent with probability proportional to its rank, and each parameter is drawn from a normal distribution around the parent's value (on a 0–1 scale, log scale for ρ, τ_min ratio and ε), with a standard deviation that starts at 0.5 and shrinks each iteration. It differs from irace in three ways: every race uses the same 30 instances in the same order (results already computed for a candidate on an instance are reused); elites can be dropped like any other candidate (irace protects them until new instances have been seen); and only numerical parameters are sampled. It was run with a budget of 1,000 (candidate, instance) evaluations, the same order of cost as the F-Race.

The iterated race was run as a second, independent search over the same space on its own seeds, to check whether a single F-Race over one sample had missed a better region.

## 5. Results

### 5.1 Budget check

The old defaults were run at six budgets on 20 instances (20 instances × 4 strategies = 80 cases per budget). "Next budget better/tied/worse" counts the cases in which the next larger budget gave a better, equal or worse plan under the ranking in Section 3. The p-value is from a two-sided Wilcoxon signed-rank test on total overdue between the budget and the next one; p < 0.05 means the difference is unlikely to be due to chance. Seconds per plan were measured with 8 plans running in parallel, so they are only comparable within this table.

| Budget (ants × iterations) | Plan evaluations | Margin | Total overdue | Overdue count | s/plan | Next budget better/tied/worse | p (total overdue, vs next) |
|---|---|---|---|---|---|---|---|
| 5 × 10 | 50 | 3.810 | 80.06 | 85.34 | 0.02 | 51/0/29 | 0.026 |
| 10 × 15 | 150 | 3.810 | 79.50 | 85.11 | 0.07 | 59/6/15 | 0.0003 |
| 10 × 30 | 300 | 3.788 | 78.74 | 84.80 | 0.14 | 50/1/29 | 0.28 |
| 20 × 30 | 600 | 3.778 | 78.63 | 84.84 | 0.31 | 44/13/23 | 0.003 |
| 20 × 60 | 1,200 | 3.772 | 78.13 | 84.38 | 0.67 | 51/3/26 | 0.008 |
| 40 × 60 | 2,400 | 3.772 | 77.83 | 84.59 | 1.30 | — | — |

Up to 1,200 evaluations, the mean margin kept falling (from 3.810 to 3.772). From 1,200 to 2,400 evaluations, the mean margin did not change and the overdue count did not improve (84.38 compared with 84.59). Total overdue fell by a further 0.3, which is statistically significant (p = 0.008) but small, while the time per plan doubled. **Chosen budget: 20 ants × 60 iterations.** This is a judgement about diminishing returns, not a proven optimum.

These numbers are identical to those of the first tuning run in the handoff document, which confirms that the reimplementation reproduces it.

### 5.2 Shared race (F-Race)

The candidates were the old defaults (candidate 0) and 32 settings from a scrambled Sobol sequence (`scipy.stats.qmc.Sobol`, seed 0) over these ranges:

| Parameter | Range |
|---|---|
| α | 0.5–2.0 |
| β | 0.5–3.0 |
| λ | 0.5–4.0 |
| ρ | 0.02–0.5 (log scale) |
| τ_min ratio | 0.001–0.2 (log scale) |
| ε | 0.01–0.3 (log scale) |

At the first test (instance 5) the Friedman test rejected strongly (p = 2.4 × 10⁻⁹) and 25 of the 33 candidates were dropped, the defaults among them. Four more were dropped at instance 6, one at instance 9 and one at instance 12. Candidates 11 and 19 were left after 12 instances and stayed statistically tied until the instances ran out (Wilcoxon p = 0.93 at instance 30). The race's pick by mean rank was candidate 11.

| Candidate | α | β | λ | ρ | τ_min ratio | ε |
|---|---|---|---|---|---|---|
| 0 (old defaults) | 1.0 | 1.0 | 2.0 | 0.1 | 0.01 | 0.05 |
| 11 | 0.585 | 2.855 | 2.595 | 0.070 | 0.0133 | 0.030 |
| 19 | 0.730 | 2.490 | 1.925 | 0.355 | 0.0240 | 0.123 |

### 5.3 Iterated race

| Iteration | Candidates raced | New candidates | Instances raced | Survivors | Elites kept | Evaluations |
|---|---|---|---|---|---|---|
| 1 | 41 | 41 | 30 | 2 | 19, 11 | 272 |
| 2 | 34 | 32 | 30 | 5 | 51, 41, 61, 19 | 321 |
| 3 | 25 | 21 | 30 | 6 | 51, 41, 61, 85 | 231 |
| 4 | 19 | 15 | 13 | 16 | 51, 108, 61, 41 | 174 |

The first iteration's 40 Sobol points start with the same 32 points as the F-Race sample, so its candidates 11 and 19 are the same settings as the F-Race's candidates 11 and 19. Those two again survived the first iteration on a different set of instances. In the last iteration the budget ran out after 13 instances, so most candidates were still tied when it stopped. The final elites were:

| Elite | α | β | λ | ρ | τ_min ratio | ε |
|---|---|---|---|---|---|---|
| 51 | 0.690 | 2.968 | 2.020 | 0.021 | 0.0017 | 0.048 |
| 108 | 0.673 | 2.753 | 1.693 | 0.024 | 0.0025 | 0.028 |
| 61 | 0.843 | 2.715 | 2.166 | 0.112 | 0.0065 | 0.112 |
| 41 | 0.674 | 2.939 | 2.028 | 0.219 | 0.0411 | 0.063 |

Both searches agree on α (about 0.6–0.85, below the old 1.0), β (about 2.5–3.0, well above the old 1.0) and λ (about 1.7–2.6, close to the old 2.0). They do not agree on ρ: the surviving settings range from ρ = 0.02 to ρ = 0.36. This means the ranking used for the race is not sensitive to ρ within that range, and the choice of ρ has to be made on other grounds (Section 5.4).

### 5.4 Confirmation on fresh instances

The old defaults, both F-Race survivors and all four irace elites were run at 20 × 60 on the 20 confirmation instances (80 instance × strategy cases each). Every setting is compared with the old defaults case by case. "Better/tied/worse" counts the cases in which the setting was lower (better), equal, or higher than the defaults on that measure; p is from a two-sided Wilcoxon signed-rank test on the 80 paired values.

| Setting | ρ | Margin | Total overdue | Overdue count | Margin better/tied/worse | Total overdue better/tied/worse (p) | Overdue count better/tied/worse (p) |
|---|---|---|---|---|---|---|---|
| Old defaults | 0.10 | 4.292 | 82.54 | 88.08 | — | — | — |
| F-Race 11 | 0.07 | 4.309 | 79.14 | 86.10 | 1/69/10 | 68/0/12 (9 × 10⁻¹¹) | 54/12/14 (1 × 10⁻⁷) |
| F-Race 19 | 0.36 | 4.319 | 78.62 | 84.79 | 0/63/17 | 70/0/10 (7 × 10⁻¹³) | 70/5/5 (6 × 10⁻¹³) |
| irace 51 | 0.02 | 4.330 | 77.88 | 84.76 | 0/55/25 | 78/0/2 (2 × 10⁻¹⁴) | 72/3/5 (3 × 10⁻¹³) |
| irace 108 | 0.02 | 4.329 | 77.86 | 84.00 | 1/54/25 | 78/0/2 (7 × 10⁻¹⁴) | 73/4/3 (4 × 10⁻¹⁴) |
| irace 61 | 0.11 | 4.321 | 78.52 | 84.53 | 0/57/23 | 75/0/5 (8 × 10⁻¹⁴) | 65/7/8 (1 × 10⁻¹²) |
| irace 41 | 0.22 | 4.316 | 78.65 | 84.91 | 1/64/15 | 73/0/7 (4 × 10⁻¹²) | 63/9/8 (2 × 10⁻¹¹) |

Every tuned setting reduced total overdue (by 3.4 to 4.7) and the overdue count (by 2.0 to 4.1) compared with the defaults, with p < 10⁻⁶ in every case. Every tuned setting was also slightly worse on margin in some cases (10 to 25 of 80) and almost never better. In those cases the margin was usually higher by 0.175 (at most 3.9% of the defaults' margin), and most of them (15 of 17 for F-Race 19) were on Camp Fire states. The mean margin rose by 0.02 to 0.04 (about 0.4–0.9%). In other words, the tuned settings leave fewer points overdue and by less, but in about a fifth of the cases the single worst point is slightly later than with the defaults.

**Change in total overdue per strategy** (tuned minus defaults; negative is better):

| Setting | ρ | Retention | Fast evaporation | Selective reset | Full reset | Largest − smallest gain |
|---|---|---|---|---|---|---|
| F-Race 11 | 0.07 | −4.41 | −3.18 | −4.08 | −1.96 | 2.45 |
| F-Race 19 | 0.36 | −4.04 | −4.15 | −3.51 | −3.98 | 0.64 |
| irace 51 | 0.02 | −5.20 | −5.04 | −4.93 | −3.50 | 1.70 |
| irace 108 | 0.02 | −5.00 | −5.15 | −4.85 | −3.72 | 1.43 |
| irace 61 | 0.11 | −4.84 | −4.42 | −4.54 | −2.29 | 2.55 |
| irace 41 | 0.22 | −4.07 | −3.92 | −3.94 | −3.64 | 0.43 |

The settings with low ρ (0.02–0.11) help the three strategies that keep some pheromone more than they help full reset, by 1.4 to 2.6. The settings with higher ρ (0.22 and 0.36) help all four strategies about equally (a spread of 0.4–0.6). An interpretation, not tested directly: with low ρ, pheromone from earlier stages lasts for many iterations, so the strategies that keep it gain more from good pheromone; with high ρ, old pheromone fades within a few iterations whatever the strategy does.

**Choice: F-Race candidate 19.** The rule set before the confirmation was to prefer a setting that is better than the defaults on fresh data and improves all four strategies about equally, so that tuning does not bias the strategy comparison. Candidates 19 and irace 41 both meet it and are nearly identical in every measure. Candidate 19 was kept because it has the lower total overdue and overdue count of the two and because it was the setting chosen in the first tuning run. irace 51 and 108 have the largest overall gains but favour the pheromone-keeping strategies over full reset, so choosing them would build part of the research question's answer into the parameters.

### 5.5 Fast-evaporation race

With the shared parameters fixed at candidate 19 (ρ = 0.36), the candidates were ρ_high ∈ {0.5, 0.7, 0.9} × k ∈ {2, 5, 10, 20}, 12 settings in all, with the current values (ρ_high = 0.5, k = 10) as candidate 0. The grid values 0.2 and 0.3 were left out because they are below ρ, so they would make the strategy evaporate more slowly after a change, not faster. Only the fast-evaporation strategy was scored.

All 12 settings survived all 30 instances; the Friedman test never rejected (p ≥ 0.07 at every test). There is therefore no evidence that any setting is better than another, and **ρ_high = 0.5 and k = 10 were kept**.

## 6. Final parameters

The values below are now the defaults in `src/aco_router/config.py` and are saved in `outputs/tuning/tuned_params.json`. The untuned values are kept in `tuning.tuner.UNTUNED_PARAMS` and `outputs/tuning/defaults_params.json`, so the untuned comparison can still be run. The shared values are candidate 19's, rounded to two or three significant figures; the confirmation ran the unrounded values.

| Parameter | Before | Tuned |
|---|---|---|
| ants × iterations | 10 × 15 (experiment scripts) | 20 × 60 |
| α | 1.0 | 0.73 |
| β | 1.0 | 2.49 |
| λ | 2.0 | 1.93 |
| ρ | 0.1 | 0.36 |
| τ_min ratio | 0.01 | 0.024 |
| ε | 0.05 | 0.12 |
| ρ_high, k | 0.5, 10 | 0.5, 10 (unchanged) |

`scripts/aco_experiment.py`, `scripts/run_change_experiment.py` and `ExperimentConfig` now default to 20 ants × 60 iterations, and both scripts accept `--params FILE.json` to run any other setting (for example the untuned one). All 146 tests pass with the tuned defaults, including the brute-force optimality test.

## 7. Findings to report

- **β changed from 1 to about 2.5.** The value β = 1 was chosen on 4-point test problems. On full-size fires, preferring nearby points gives better plans. Both independent searches put β between 2.5 and 3.0. The progress report's statement that β = 1 was chosen deliberately should be updated.
- **The tuned setting is better on overdue measures and slightly worse on margin.** On fresh instances, total overdue fell by 3.9 (better in 70 of 80 cases) and the overdue count by 3.3 (better in 70 of 80), while the margin was equal in 63 cases and worse in 17, by about 0.6% on average.
- **ρ is not determined by the race, but it changes how much the strategies differ.** Settings from ρ = 0.02 to ρ = 0.36 were statistically tied in both searches, yet low-ρ settings favour the pheromone-keeping strategies over full reset in the confirmation. The chosen ρ = 0.36 treats the strategies equally, which keeps the comparison fair, but it also means old pheromone fades quickly. The strategy experiments should be read with this in mind, and it may be worth repeating the main strategy comparison at a low ρ as a sensitivity check.

## 8. Reproducing

```sh
python scripts/tune_aco.py budget --camp --instances 20 --workers 8 --out outputs/tuning/budget
python scripts/tune_aco.py shared --camp --ants 20 --iterations 60 --instances 30 --workers 9 --out outputs/tuning/shared
python scripts/tune_aco.py irace --camp --ants 20 --iterations 60 --instances 30 --budget 1000 --workers 9 --out outputs/tuning/irace
python scripts/tune_aco.py confirm defaults=defaults frace11=outputs/tuning/shared_candidate11.json \
    frace19=outputs/tuning/shared_candidate19.json irace51=outputs/tuning/irace_elite51.json \
    irace108=outputs/tuning/irace_elite108.json irace61=outputs/tuning/irace_elite61.json \
    irace41=outputs/tuning/irace_elite41.json --camp --workers 9 --out outputs/tuning/confirm
python scripts/tune_aco.py fast-evaporation --params outputs/tuning/chosen_shared.json --camp --ants 20 \
    --iterations 60 --instances 30 --workers 9 --out outputs/tuning/fast_evaporation
```

On a 10-core machine each step took between 1 and 15 minutes (the iterated race was the longest). Without `--params`, the tuning steps start from the untuned parameters, so they give the same results after `config.py` was updated.

Output files, per race: `<out>_candidates.csv` (each candidate's parameters, when it was dropped, and whether it survived), `<out>_log.csv` (one row per instance: candidates alive, test used, p-value, candidates dropped), `<out>_results.csv` (every plan's scores) and `<out>_params.json` / `<out>_candidate<N>.json` (the pick and the survivors). The iterated race writes `_iterations.csv` and `_elite<N>.json` instead of `_log.csv` and `_candidate<N>.json`. The confirmation writes the raw results, `_summary.csv` (paired comparisons overall and per strategy) and `_means.csv`.

## References

- Birattari, M., Stützle, T., Paquete, L. and Varrentrapp, K. (2002). A racing algorithm for configuring metaheuristics. *GECCO 2002*.
- López-Ibáñez, M., Dubois-Lacoste, J., Pérez Cáceres, L., Birattari, M. and Stützle, T. (2016). The irace package: iterated racing for automatic algorithm configuration. *Operations Research Perspectives*, 3, 43–58.
- Stützle, T. and Hoos, H. H. (2000). MAX–MIN Ant System. *Future Generation Computer Systems*, 16(8), 889–914.
