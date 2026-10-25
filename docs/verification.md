# Verification record (25 October 2026)

- Environment: Python 3.11. Versions are in `requirements.lock`.
- `pytest -q`: 50 tests pass. The tests do not use the network.
- Ground-truth fire, fire 3885, first 8 snapshots: raster 347 × 377 cells at 100 m. All 8 snapshots agree exactly with FEDS.
- Mission, fire 3885, 24 h, 6 drones, 1 km cells, `smoke_router`: 144 stages. Median map points: 138. Median changed fraction: 0.056. Mean `front_on_map_fraction`: 0.95.
- Validation: refer to the table in [continuous-fire.md](continuous-fire.md).

`smoke_router` only tests the pipeline. The ACO, greedy and VNS planners and the experiments are not done.
