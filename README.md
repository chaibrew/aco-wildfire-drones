# aco-wildfire-drones

ACO routing for wildfire-monitoring drones (COMP3821). The project compares pheromone strategies when the fire perimeter changes.

## Setup

Use Python 3.11 or later.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
pytest -q
```

`requirements.lock` has the exact versions. `pyproject.toml` has the permitted version ranges.

## Data

```sh
python scripts/fetch_feds.py --year 2018
fire-sim inspect data/raw/LargeFires_2018.gpkg
```

The download script gets the FEDS archive (approximately 99 MB) one time, verifies the checksum, and extracts one year. `inspect` shows the layers, the fire IDs and the number of snapshots for each fire.

## Ground-truth fire

FEDS gives one perimeter for a fire approximately each 12 hours. `fire-sim fire` makes a continuous fire from these perimeters. The result is an arrival time for each 100 m cell.

- At each FEDS observation time, the fire agrees with the FEDS perimeter.
- Between observations, wind, slope and hourly fire weather control the spread.
- The command downloads weather (Open-Meteo, ERA5) and elevation (Copernicus GLO-30) one time for each fire. It keeps them in `data/cache/`.

```sh
fire-sim fire data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8 --gif \
  --output outputs/fire-3885
```

| Option | Function |
| --- | --- |
| `--seed`, `--noise` | Change the random spread variation. All seeds agree with the FEDS perimeters. |
| `--no-weather`, `--no-terrain` | Remove weather or slope from the spread. |
| `--pacing-exponent` | Set how strongly fire weather controls the hourly growth. Default 2. |
| `--validate` | Do the leave-one-snapshot-out check. |

Output: `field.npz` and `field.json` (the fire), `fire.png`, `fire.gif` (with `--gif`), `validation.csv` (with `--validate`).

## Drone mission

`fire-sim mission` flies drones above the ground-truth fire. The planner does not see the ground truth. The planner sees only the fleet map:

- The most recent FEDS perimeter. A new perimeter arrives each 12 hours, with an optional delay.
- Drone observations. When a drone completes an observation, it sees the true fire in its sensor radius.

The monitoring points are the grid cells on the mapped perimeter. The risk of a point comes from the observed spread rate. For each stage, the mission calculates J two times: on the fleet map and on the true active front.

```sh
fire-sim mission data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8 \
  --duration-h 24 --drones 6 --cell-size 1000 --gif --output outputs/mission-3885
```

| Option | Function | Default |
| --- | --- | --- |
| `--cell-size` | Monitoring grid cell (m) | 500 |
| `--sensor-radius` | Drone sensor radius (m) | cell size |
| `--snapshot-latency-h` | Delay before a FEDS perimeter arrives (h) | 0 |
| `--rate-tiers LOW HIGH` | Spread rate (m/min) for risk 2 and risk 3 | 2 10 |
| `--revisit` | Base revisit period q (s). q = base / risk. | 900 |
| `--horizon` | Time between replans (s) | 600 |
| `--drones`, `--speed`, `--service` | Fleet size, speed (m/s), observation time (s) | 3, 15, 5 |

Output: `metrics.csv` (one row for each stage), `config.json`, `mission.png`, `mission.gif` (with `--gif`).

## Connect a planner

A planner is a function. It receives a `PlanningState` and returns `{drone_id: [cell_id, ...]}`. Use `evaluate(state, routes)` to calculate J for a candidate. Refer to [the simulation contract](docs/simulation-contract.md) and [scripts/continuous_example.py](scripts/continuous_example.py).

`smoke_router` only tests the pipeline. It is not a baseline.

## Model details

Refer to [docs/continuous-fire.md](docs/continuous-fire.md).

## Legacy snapshot replay

`fire-sim demo` and `fire-sim feds` replay the FEDS perimeters without interpolation. They compress time (default 72 times). The planner sees each perimeter directly. These commands do not agree with the mission model. They contain the controlled-growth tool:

```sh
fire-sim demo --static --steps 6 --perturb-stage 2 --locality local --radius 400 --patches 4 \
  --seed 42 --output outputs/local-change
```

`--locality local` adds growth patches on a short part of the boundary. `--locality distributed` adds them along all of the boundary. The radius and the number of patches control the geometry, not the fraction of changed points. Read the fraction in `metrics.csv` (`affected_union_fraction`).

## Data sources

- [Chen et al. (2022), FEDS paper](https://doi.org/10.1038/s41597-022-01343-0)
- [FEDS collection, version 1](https://doi.org/10.6084/m9.figshare.c.5601537.v1)
- [Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1) (CC0)
- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/)

The loader uses the `perimeter` layer of the published 2012–2020 schema. Other FEDS releases can have different field names.
