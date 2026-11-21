# aco-wildfire-drones

ACO routing for wildfire-monitoring drones (COMP3821). This project compares different pheromone strategies when the fire perimeter changes.

See it in action! (recommend viewing on desktop): [aco-wildfire-drones.vercel.app](https://aco-wildfire-drones.vercel.app)

## Setup

Use Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
pytest -q
```

## Data

The repository contains the FEDS 2018 GeoPackage (`data/raw/`) and the weather and elevation for the tested fires (`data/cache/weather/`, `data/cache/dem/`). For another year, download it:

```sh
python scripts/fetch_feds.py --year 2019
fire-sim inspect data/raw/LargeFires_2019.gpkg
```

## Ground-truth fire

FEDS tracks fire perimeters at 12 hour intervals. `fire-sim fire` makes a continuous fire by interpolating between 2 12-hour timestamp pairs with augmentation with weather/terrain data to try to make it somewhat realistic.

- At each FEDS observation time, the fire perimeter is exactly the same as the FEDS perimeter.
- Between observations, wind, slope and hourly fire weather control the spread.
- The command downloads weather (Open-Meteo, ERA5) and elevation (Copernicus GLO-30) one time for each fire. It keeps them in `data/cache/`.

```sh
fire-sim fire configs/camp-3885.toml
```

Output in `[output] fire_dir`: `field.npz` and `field.json` (the fire), `fire.png`, `fire.gif` (if `gif = true`), `validation.csv` (if `validate = true`), and a copy of the config.

## Drone mission

`fire-sim mission` flies drones above the ground-truth fire. The planner does not see the ground truth. The planner sees:

- The most recent FEDS perimeter. A new perimeter arrives each 12 hours.
- Drone observations. When a drone completes an observation, it sees the true fire in its sensor radius.

The monitoring points are the grid cells on the mapped perimeter. The risk of a point comes from the observed spread rate. The planner objective is J on the fleet map.

The mission runs in 12-hour blocks. A block starts at one FEDS snapshot and ends at the next. At the end of a block, the next snapshot first scores the fleet map. Then the planner receives it as the start of the next block.

```sh
fire-sim mission configs/camp-3885.toml
```

Output in `[output] mission_dir`: `metrics.csv` (one row for each stage), `blocks.csv` (one row for each 12-hour block, with the snapshot score), `run.json` (the full config and the fire provenance), `mission.png`, `mission.gif` (if `gif = true`), and a copy of the config.

## Configuration

Both commands read one TOML file. Copy [configs/camp-3885.toml](configs/camp-3885.toml) and change the values. If you remove a key, the default applies. An unknown key causes an error.

| Section | Key | Default | Function |
| --- | --- | --- | --- |
| `[data]` | `path` | required | FEDS GeoPackage |
| | `fire_id` | required | FEDS fire ID |
| | `year` | none | Necessary if the fire ID occurs in more than one year |
| | `limit` | all | Use only the first N snapshots |
| | `crs` | UTM from first perimeter | Metric CRS, for example `"EPSG:32610"` |
| `[fire]` | `res_m` | 100.0 | Fire grid cell (m) |
| | `seed`, `noise_sigma` | 0, 0.3 | Random spread variation. 0 removes it. |
| | `weather`, `terrain` | true, true | Use weather or slope in the spread |
| | `time_convention` | `"overpass"` | `"overpass"`: satellite times. `"label"`: FEDS labels as UTC. |
| `[mission]` | `cell_size_m` | 500.0 | Monitoring grid cell (m) |
| | `sensor_radius_m` | cell size | Drone sensor radius (m) |
| | `base_revisit_s` | 900.0 | Base revisit period. q = base / risk. |
| | `rate_tiers_mpm` | [2.0, 10.0] | Spread rate (m/min) for risk 2 and risk 3 |
| | `drones`, `speed_mps`, `service_s` | 3, 15.0, 5.0 | Fleet size, speed (m/s), observation time (s) |
| | `horizon_s` | 600.0 | Time between replans (s) |
| `[run]` | `start_h` | 0.0 | Mission start, hours after the first snapshot |
| | `duration_h` | to last snapshot | Mission length (h) |
| `[output]` | `fire_dir`, `mission_dir` | `outputs/fire`, `outputs/mission` | Output folders |
| | `gif` | false | Make animations |
| | `gif_step_h`, `gif_every` | 0.5, 3 | Fire frame step (h), mission stages for each frame |
| | `validate` | false | Do the leave-one-snapshot-out check (`fire` only) |

## Connect a planner

A planner is a function. It receives a `PlanningState` and returns `{drone_id: [cell_id, ...]}`. Use `evaluate(state, routes)` to calculate J for a candidate. Refer to [the simulation contract](docs/simulation-contract.md) and [scripts/continuous_example.py](scripts/continuous_example.py).

`smoke_router` only tests the pipeline. It is not a baseline.

## ACO router

`src/aco_router` is the ACO planner with the four pheromone strategies: retention, fast evaporation, selective reset and full reset. `FireSimRouter` connects it to `run_mission`. It scores each candidate plan with `evaluate` and moves pheromone by `CellId`.
## ACO router
`src/aco_router` is the ACO planner with the four pheromone strategies: retention, fast evaporation, selective reset and full reset. `FireSimRouter` connects it to `run_mission`. It scores each candidate plan with `evaluate` and moves pheromone by `CellId`.

The experiment needs a FEDS Largefire GeoPackage ([Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1), years 2012–2020) in `data/raw/`. `python scripts/fetch_feds.py --year 2018` downloads one. The defaults are `LargeFires_2018.gpkg` and fire 3885 (the Camp Fire). To use a different file or fire, add `--gpkg`, `--year` and `--fire-id`. The first run for a new fire downloads its weather and elevation into `data/cache/`.

- `aco_experiment.py` flies `smoke_router` and the four ACO strategies over the Camp Fire (fire 3885, first 24 h, 144 replans of 10 minutes) for 10 seeds. Each seed changes the fire between snapshots and the random choices of the ACO. The script writes one row for each run, with map J, true-front J and overdue counts. It takes approximately 30 to 60 minutes.
- `--help` shows the config options (fire, snapshots, hours, seeds, drones, ACO budget).
- `aco_analyse.py` reads that CSV and compares each strategy with `smoke_router` on the same seed. For each metric it gives the mean difference, a paired t statistic and the number of seeds that each strategy won.

For example, the commands below run the tests, then runs `smoke_router` and the four ACO strategies with 6 drones over 10 seeds, and writes the results to a file at `outputs/camp3885-6-drones.csv`. The last command compares each strategy with `smoke_router` on the same seed. For each metric it gives the mean difference, a paired t statistic (with 10 seeds, |t| > 2.26 means p < 0.05) and the number of seeds where the strategy was better.


```sh
python -m pytest -q
python scripts/aco_experiment.py --drones 6 --out outputs/camp3885-6-drones.csv
python scripts/aco_analyse.py outputs/camp3885-6-drones.csv
```

The ACO parameters are in `src/aco_router/config.py`.

## Model details

Refer to [docs/continuous-fire.md](docs/continuous-fire.md).

## Data sources

- [Chen et al. (2022), FEDS paper](https://doi.org/10.1038/s41597-022-01343-0)
- [FEDS collection, version 1](https://doi.org/10.6084/m9.figshare.c.5601537.v1)
- [Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1) (CC0)
- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/)

The loader uses the `perimeter` layer of the published 2012–2020 schema. Other FEDS releases can have different field names.


## Baselines and controlled-change experiments

Greedy and VNS planners (`src/baselines`) and the paired controlled-change experiment (`src/experiment`,
`scripts/run_change_experiment.py`, `scripts/analyse_change_experiment.py`) are described in
[docs/experiment-harness.md](docs/experiment-harness.md).
