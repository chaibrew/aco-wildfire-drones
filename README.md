# aco-wildfire-drones

ACO routing for wildfire-monitoring drones (COMP3821, Group 17). The project compares four pheromone strategies when the fire perimeter changes: retention, fast evaporation, selective reset and full reset. Greedy and VNS are the baselines.

Demo site (use a desktop browser): [aco-wildfire-drones.vercel.app](https://aco-wildfire-drones.vercel.app)

## Setup

Use Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
pytest -q
```

The tests do not use the network. 169 tests pass.

## Repository

| Folder | Contents |
| --- | --- |
| `src/fire_sim` | Ground-truth fire, drone mission and planning state |
| `src/aco_router` | ACO planner and the four pheromone strategies. The parameters are in `config.py`. |
| `src/baselines` | Greedy and VNS |
| `src/experiment` | Controlled-change experiment and statistics |
| `src/tuning` | F-Race and iterated race |
| `scripts` | Commands that run the experiments and make the tables |
| `outputs`, `results` | Result files that the report uses |
| `docs` | Model and method details |
| `site` | Demo site. Refer to [site/README.md](site/README.md). |

## Reproduce the report

Run each command from the repository root. Each command writes its files to `outputs/`.

| Report section | Command | Result files in the repository |
| --- | --- | --- |
| Tuning (3.3.4, Appendix A) | Refer to [docs/tuning.md](docs/tuning.md), section 8 | Tables in `docs/tuning.md` |
| Controlled change, main run (6.1) | `scripts/controlled_change_rerun.sh tuned` | `results/cc_tuned_rho01_results.zip` |
| Controlled change, baselines (6.1) | `scripts/controlled_change_rerun.sh baselines` | `outputs/cc/camp_rho01_baselines.csv` |
| Controlled change, before tuning (6.3) | `scripts/controlled_change_rerun.sh untuned` | `cc/camp_untuned_cycle.csv` in `results/cc_mission_results.zip` |
| 24 h replay (6.2) | `python scripts/run_tuned_missions.py outputs/tuning_missions_d10 --baselines` | `outputs/tuning_missions_d10/` |

To get the statistics:

```sh
python scripts/analyse_change_experiment.py outputs/cc/camp_tuned_rho01_cycle.csv
python scripts/analyse_tuned_missions.py outputs/tuning_missions_d10
```

To compare greedy and VNS with full reset, join the two controlled-change files first:

```sh
(cat outputs/cc/camp_tuned_rho01_cycle.csv; tail -n +2 outputs/cc/camp_rho01_baselines.csv) > outputs/cc/camp_all.csv
python scripts/analyse_change_experiment.py outputs/cc/camp_all.csv --group all --no-interaction
```

All runs use fixed seeds. The controlled-change runs use seeds 5000–5009 and the replay uses seeds 0–9. The tuning seeds do not overlap them.

## Data

The repository contains the FEDS 2018 GeoPackage (`data/raw/`) and the weather and elevation for the tested fires (`data/cache/`). For another year, download it:

```sh
python scripts/fetch_feds.py --year 2019
fire-sim inspect data/raw/LargeFires_2019.gpkg
```

## Ground-truth fire

FEDS gives a fire perimeter each 12 hours. `fire-sim fire` makes a continuous fire between these perimeters.

- At each FEDS time, the fire is the same as the FEDS perimeter.
- Between FEDS times, wind, slope and hourly fire weather control the spread.
- The command downloads weather (Open-Meteo, ERA5) and elevation (Copernicus GLO-30) one time for each fire. It keeps them in `data/cache/`.

```sh
fire-sim fire configs/camp-3885.toml
```

The output is in `[output] fire_dir`: `field.npz` and `field.json` (the fire), `fire.png`, `fire.gif` (if `gif = true`), `validation.csv` (if `validate = true`) and a copy of the config.

## Drone mission

`fire-sim mission` flies drones above the ground-truth fire. The planner does not see the ground truth. The planner sees:

- The most recent FEDS perimeter. A new perimeter arrives each 12 hours.
- Drone observations. A drone sees the true fire in its sensor radius.

The monitoring points are the grid cells on the mapped perimeter. The risk of a point comes from the observed spread rate. The planner objective is J on the fleet map.

```sh
fire-sim mission configs/camp-3885.toml
```

The output is in `[output] mission_dir`: `metrics.csv` (one row for each stage), `blocks.csv` (one row for each 12-hour block), `run.json`, `mission.png`, `mission.gif` (if `gif = true`) and a copy of the config.

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

A planner is a function. It receives a `PlanningState` and returns `{drone_id: [cell_id, ...]}`. Use `evaluate(state, routes)` to calculate J for a plan. Refer to [docs/simulation-contract.md](docs/simulation-contract.md) and [scripts/continuous_example.py](scripts/continuous_example.py).

`smoke_router` only tests the pipeline. It is not a baseline.

## Documents

- [docs/continuous-fire.md](docs/continuous-fire.md): fire model and validation
- [docs/simulation-contract.md](docs/simulation-contract.md): planner interface
- [docs/experiment-harness.md](docs/experiment-harness.md): baselines and controlled-change experiment
- [docs/tuning.md](docs/tuning.md): parameter tuning

## Data sources

- [Chen et al. (2022), FEDS paper](https://doi.org/10.1038/s41597-022-01343-0)
- [FEDS collection, version 1](https://doi.org/10.6084/m9.figshare.c.5601537.v1)
- [Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1) (CC0)
- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/)

The loader uses the `perimeter` layer of the published 2012–2020 schema. Other FEDS releases can have different field names.
