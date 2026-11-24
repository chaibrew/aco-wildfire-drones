# Verification record

## 24 November 2026

- Environment: Python 3.11. The versions are in `requirements.lock`.
- `pytest -q`: 169 tests pass. The tests do not use the network.
- `scripts/analyse_change_experiment.py` on the main run and the baseline file gives the greedy and VNS margin differences in report section 6.1 (+0.446 and +0.222).
- A rerun of `scripts/controlled_change_rerun.sh baselines` for stage 6, seed 5000 gives the same values as `outputs/cc/camp_rho01_baselines.csv` in 19 of 20 rows. In the other row, VNS has the same margin and a different total overdue.
- Demo site: `npm run typecheck` and `npm run build` pass.

## 25 October 2026

- Ground-truth fire, fire 3885, first 8 snapshots: raster 347 × 377 cells at 100 m. All 8 snapshots agree exactly with FEDS.
- Validation: refer to the table in [continuous-fire.md](continuous-fire.md).
