# aco-wildfire-drones

ACO routing for wildfire-monitoring drones (COMP3821). This project compares different pheromone strategies when the fire perimeter changes.

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

```sh
python scripts/fetch_feds.py --year 2018
fire-sim inspect data/raw/LargeFires_2018.gpkg
```

## Ground-truth fire

FEDS tracks fire perimeters at 12 hour intervals. `fire-sim fire` makes a continuous fire by interpolating between 2 12-hour timestamp pairs with augmentation with weather/terrain data to try to make it somewhat realistic.

- At each FEDS observation time, the fire perimeter is exactly the same as the FEDS perimeter.
- Between observations, wind, slope and hourly fire weather control the spread.
- The command downloads weather (Open-Meteo, ERA5) and elevation (Copernicus GLO-30) one time for each fire. It keeps them in `data/cache/`.

```sh
fire-sim fire data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8 --gif \
  --output outputs/fire-3885
```

| Option | Function |
| --- | --- |
| `--seed`, `--noise` | Change the random spread variation. |
| `--no-weather`, `--no-terrain` | Remove weather or slope from the spread. |
| `--validate` | Do a leave-one-snapshot-out check. |

Output: `field.npz` and `field.json` (the fire), `fire.png`, `fire.gif` (with `--gif`), `validation.csv` (with `--validate`).

## Drone mission

`fire-sim mission` flies drones above the ground-truth fire. The planner does not see the ground truth. The planner sees:

- The most recent FEDS perimeter. A new perimeter arrives each 12 hours.
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

## Data sources

- [Chen et al. (2022), FEDS paper](https://doi.org/10.1038/s41597-022-01343-0)
- [FEDS collection, version 1](https://doi.org/10.6084/m9.figshare.c.5601537.v1)
- [Largefire archive](https://figshare.com/articles/dataset/Largefire_tar_gz/19170071/1) (CC0)
- [Open-Meteo historical weather](https://open-meteo.com/en/docs/historical-weather-api) (ERA5)
- [Copernicus GLO-30 DEM](https://registry.opendata.aws/copernicus-dem/)

The loader uses the `perimeter` layer of the published 2012–2020 schema. Other FEDS releases can have different field names.
