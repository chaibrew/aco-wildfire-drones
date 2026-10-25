# aco-wildfire-drones

Ant Colony Optimisation for multi-drone wildfire monitoring, comparing pheromone strategies under dynamic fire perimeters (COMP3821).

## Fire-perimeter simulation and FEDS integration

Runnable Python foundation for the COMP3821 drone-monitoring project. It reads historical FEDS perimeters, creates stable monitoring points, executes routes, and records change sets and revisit metrics. The ACO variants, final greedy/VNS baselines, and experimental sweep are separate components to plug into this environment.

### Quick start

The project already has a `.venv` and a downloaded 2018 FEDS GeoPackage. From this folder:

```sh
source .venv/bin/activate
fire-sim demo --output outputs/demo
fire-sim demo --static --output outputs/static
pytest -q
```

For a fresh machine (Python 3.11 or newer):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
```

`requirements.lock` pins the versions used during setup. `pyproject.toml` contains the supported dependency ranges. GIS dependencies install from wheels on supported platforms. No separate QGIS application or fire-spread engine is needed.

### Continuous fire and drone missions

FEDS only observes the fire every ~12 hours. `fire-sim fire` turns a FEDS sequence into a continuous **ground-truth** fire: an arrival time for every 100 m cell. The gaps between snapshots are filled by shortest-time spread through a wind/slope rate field, paced hour by hour by fire weather. Every observed snapshot is reproduced exactly. Weather (Open-Meteo/ERA5) and elevation (Copernicus GLO-30) are downloaded once per fire into `data/cache/`.

`fire-sim mission` flies drones over that fire, but **planners never see the ground truth**. They see only the fleet's own map:
- the latest FEDS snapshot, which arrives every ~12 hours (optionally delayed);
- whatever drones observe within their sensor radius when they visit a point.

Monitoring points are the cells on the mapped perimeter. Risk comes from observed spread: growth in the last snapshot interval, or advance a drone measured since the area was last seen. Each stage is scored twice: J on the map (what the planner optimised) and J on the true active front (what actually needed watching). The run also reports how much of the true front is on the map.

See [the model write-up](docs/continuous-fire.md).

```sh
# Build, plot and animate 4 days of the Camp Fire (ground truth).
fire-sim fire data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8 --gif \
  --output outputs/fire-3885

# 24 h mission: 6 drones replanning every 600 s, 1 km cells, 1 km sensor radius.
fire-sim mission data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8 \
  --duration-h 24 --drones 6 --cell-size 1000 --gif --output outputs/mission-3885
```

**Fire options:** `--seed` and `--noise` (different but snapshot-consistent fires), `--no-weather`, `--no-terrain`, `--pacing-exponent`, `--validate` (leave-one-snapshot-out check).

**Mission options:** `--sensor-radius`, `--snapshot-latency-h`, `--rate-tiers LOW HIGH` (m/min thresholds for risk tiers 2 and 3), `--horizon`, `--drones`, `--speed`.

[scripts/continuous_example.py](scripts/continuous_example.py) shows the planner hook.

### Run real FEDS data (snapshot replay)

```sh
# Downloads ~99 MB once, verifies checksum, extracts only the requested year.
python scripts/fetch_feds.py --year 2018

# Lists layers, columns, available fire IDs and snapshot counts.
fire-sim inspect data/raw/LargeFires_2018.gpkg

# Verified example: eight snapshots of fireID 3885 in 2018.
fire-sim feds data/raw/LargeFires_2018.gpkg \
  --year 2018 --fire-id 3885 --steps 8 \
  --output outputs/feds-2018-3885
```

The loader uses the `perimeter` layer, selects `(year, fireID)`, sorts `time`, and chooses one UTM projection from the first perimeter. All distances thereafter are metres. Use `--crs EPSG:32610` to specify a metric projection explicitly. Invalid geometry fails visibly unless `--repair` is supplied; repair counts are recorded.

FEDS has half-daily source snapshots. By default, `--time-scale 72` replays 12 source hours as 600 simulation seconds. This is an experimental time compression, **not observed fire speed**. Use `--time-scale 1` for source elapsed time. Between snapshots the perimeter is held constant. The last snapshot runs for `--horizon` seconds (default 600); `--static` freezes the first snapshot and uses that horizon at every stage. Source half-day labels retain their unspecified timezone.

Each output folder contains:

| File | Contents |
| --- | --- |
| `scenario.json` | Source SHA-256 and CRS, configuration, timestamped geometries, stable point IDs, synthetic risk/revisit values, initial state at each stage, change sets, routes, observation events |
| `metrics.csv` | Per-stage overdue objective, overdue count, mean/max gaps, route distances, changed fraction, planning time |
| `preview.png` | First/last perimeters and monitoring points, point-count evolution, overdue trace |

The JSON geometry coordinates are in the recorded projected CRS; this is a scenario format, not RFC 7946 geographic GeoJSON. Large raw data, generated outputs and virtual environments are excluded by `.gitignore`.

### Controlled perimeter changes

Start with a static shape to isolate one controlled change from historical evolution:

```sh
fire-sim demo --static --steps 6 --perturb-stage 2 \
  --locality local --radius 400 --patches 4 --seed 42 \
  --output outputs/local-change

fire-sim demo --static --steps 6 --perturb-stage 2 \
  --locality distributed --radius 400 --patches 4 --seed 42 \
  --output outputs/distributed-change
```

These options also work with `feds --static` to perturb a real starting perimeter. Circular growth patches are concentrated along a short boundary arc or distributed along exterior boundaries. The perturbation applies from the selected stage onwards. On moving historical perimeters the same seeded boundary-relative rule is reapplied to each geometry; this is a controlled overlay, not a physical growth model.

Radius and patch count control geometry. They do **not** guarantee an exact affected-point fraction or identical magnitude across locality conditions. Record the realised `affected_union_fraction`, then calibrate and freeze parameters on pilot instances before the final comparisons. The current generator implements growth; controlled shrinkage and exact target-fraction calibration are future extensions.

### Hook up the algorithms

See [the simulation contract](docs/simulation-contract.md) and [the runnable integration example](scripts/router_example.py). Every planner receives a `PlanningState` and returns a dictionary `{drone_id: [cell_id, ...]}`. Cell IDs are `(grid_x, grid_y)` tuples. Repeated visits are allowed. Evaluate candidate solutions with the pure `evaluate(state, routes)` function; commit the selected solution with `env.step(routes, horizon_s)`.

`smoke_router` exists to check the end-to-end pipeline. It is not the group's final greedy baseline, and its output is not an ACO result. Risk values are synthetic stable assignments; FEDS does not supply the project's monitoring priorities. The defaults (500 m cells, 3 drones, 15 m/s, 5 s service, revisit periods 300/450/900 s) are starting values requiring pilot calibration. Fire 3885 produces hundreds of points and substantial overdue time with these defaults.

### Data sources

- [Chen et al. (2022), FEDS paper](https://doi.org/10.1038/s41597-022-01343-0).
- [Published FEDS collection, version 1](https://doi.org/10.6084/m9.figshare.c.5601537.v1).
- [Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1), published under CC0 according to its Figshare metadata.

The downloader pins the archive's article version and writes a provenance manifest beside the GeoPackage. It does not use the serialization/pickle products. The adapter targets this published historical schema; other FEDS/FireAtlas releases may require field mapping.
