# Simulation contract

All planners use this interface. Units: metres and seconds. Coordinates: the projected CRS of the fire (UTM).

## Data flow

```text
load_feds -> load_fire -> ArrivalField (hidden ground truth)
                              |
FEDS snapshots + drone observations -> FireMap -> PlanningState -> router -> Routes
                                                                               |
                                    FireMap.observe <- visits <- Environment.step
```

## Types

```python
CellId = tuple[int, int]                 # (floor(x / cell_size_m), floor(y / cell_size_m)), origin (0, 0)
Routes = dict[int, list[CellId]]         # drone id -> ordered point IDs

@dataclass(frozen=True)
class MonitoringPoint:
    id: CellId
    x: float                             # cell centre (m)
    y: float
    risk: int                            # 1, 2 or 3
    revisit_s: float                     # q = base_revisit_s / risk

@dataclass(frozen=True)
class Drone:
    id: int
    x: float                             # current position (m)
    y: float

@dataclass(frozen=True)
class PlanningState:                     # planner input; do not change it
    time_s: float                        # stage start t
    horizon_s: float                     # H
    speed_mps: float
    service_s: float                     # observation time for each visit
    drones: tuple[Drone, ...]
    points: dict[CellId, MonitoringPoint]
    last_seen: dict[CellId, float]       # last observation, or activation time
    coverage: dict[CellId, tuple[CellId, ...]] | None   # points that one visit observes (sensor radius, with itself)
```

## Planner

```python
def router(state: PlanningState) -> Routes: ...
```

| Rule | Detail |
| --- | --- |
| Keys | One key for each `state.drones[i].id`. A list can be empty. |
| Items | Each ID must be in `state.points`. Repeats are permitted. |
| Time | For each drone: `sum(leg_m / speed_mps + service_s) <= horizon_s`. |
| Start | The first leg starts at the drone position. There is no return to base. |
| End | A drone waits at its last point. It does not observe while it waits. |

`evaluate` and `Environment.step` raise `ValueError` if a rule is broken. They do not cut a route.

## Evaluation

```python
def evaluate(state: PlanningState, routes: Routes) -> Evaluation: ...   # does not change the state

@dataclass
class Evaluation:
    objective: float                             # J
    overdue_count: int                           # points with P_i > q_i
    mean_gap_s: float                            # mean P_i
    max_gap_s: float                             # max P_i
    total_distance_m: float
    longest_route_m: float
    gaps_s: dict[CellId, float]                  # P_i
    observations: dict[CellId, list[float]]      # sorted, no duplicates
    end_drones: tuple[Drone, ...]
    visits: list[tuple[float, int, CellId]]      # (time, drone_id, visited point), sorted by time
```

```text
visit time   = arrival time + service_s
a visit to i observes every point in coverage[i] (only i if coverage is None)
times_i      = [last_seen[i], observations of i in this stage, t + H]
P_i          = largest difference of adjacent values in times_i
J            = max_i max(0, P_i / q_i - 1)          (0 if there are no points)
```

- A gap can start before the stage. Overdue time continues across replans. State this extension of the proposal objective in the report.
- Activation is not an observation. A new point starts at its activation time.

## Changes

```python
def changes(previous: dict[CellId, MonitoringPoint],
            current: dict[CellId, MonitoringPoint]) -> ChangeSet: ...

@dataclass(frozen=True)
class ChangeSet:
    added: frozenset[CellId]                     # in current only
    removed: frozenset[CellId]                   # in previous only
    changed: frozenset[CellId]                   # in both; risk or revisit_s is different
    affected_current: frozenset[CellId]          # added | changed = C_t (property)
    def fraction(self, previous, current) -> float: ...   # |added ∪ removed ∪ changed| / |previous ∪ current|
```

Pheromone rules:

- Use `sorted(state.points)` for matrix indices. The indices are correct for one stage only.
- Move pheromone by `CellId`, not by matrix index.
- New edges start at `tau_0`. Delete all edges of `removed` points, also when `C_t` is empty.

## Environment

```python
env = Environment(drones: list[Drone], *, speed_mps=15.0, service_s=5.0, sensor_radius_m=0.0)
env.activate(points: dict[CellId, MonitoringPoint]) -> None   # new IDs: last_seen = env.time_s
                                                              # kept IDs: last_seen does not change
                                                              # missing IDs: removed with their history
env.state(horizon_s: float) -> PlanningState
env.step(routes: Routes, horizon_s: float) -> Evaluation       # updates last_seen and drones; time_s += H
env.mark_observed(time_s: float, ids=None) -> None             # a snapshot observes all (or ids) points
```

## Mission

```python
config = MissionConfig(cell_size_m=500.0, sensor_radius_m=None,   # None: radius = cell size
                       base_revisit_s=900.0, rate_tiers_mpm=(2.0, 10.0),
                       drones=3, speed_mps=15.0, service_s=5.0, horizon_s=600.0)

rows = run_mission(field: ArrivalField, router, config,
                   *, start_s=None, stop_s=None,             # default: first and last snapshot
                   record=None) -> list[dict]               # record(stage, fire_map, points, routes)
```

One row for each stage:

| Key | Value |
| --- | --- |
| `stage`, `time_s`, `horizon_s` | Stage number, start t, H |
| `snapshot_delivered` | Snapshot indices given to the planner at this stage |
| `map_points` | Number of points on the fleet map |
| `added`, `removed`, `changed`, `affected_current`, `affected_union_fraction` | `changes()` results |
| `planning_s` | Router time (s) |
| `visits` | Completed observations |
| `map_objective`, `map_overdue_count` | J and overdue count on the fleet map |
| `true_front_points`, `true_objective`, `true_overdue_count`, `true_max_gap_s` | The same on the true active front |
| `front_on_map_fraction` | True front cells on the map / true front cells |
| `known_burned_fraction` | Known burned area / true burned area |
| `total_distance_m`, `longest_route_m` | Route lengths |

## Loading

```python
sequence, provenance = load_feds(path, *, fire_id, year=None, target_crs=None,
                                 repair=False, limit=None)
# sequence:   list[Perimeter(timestamp: datetime, geometry: Polygon | MultiPolygon)], time order, metric CRS
# provenance: {"source", "source_sha256", "source_crs", "target_crs", "year", "fire_id", "layer",
#              "geometry_repairs", "timestamp_convention", "duplicate_rows_unioned", "dataset_doi"}

field, sequence = load_fire(path, *, fire_id, year=None, limit=None, crs=None, repair=True,
                            cache_dir="data/cache", res=100.0, pad=2000.0, seed=0,
                            model=SpreadModel(), weather=True, terrain=True, convention="overpass")
```

Provenance records where the data came from, so that another person can repeat a run. `fire-sim` writes it into `field.json` and `config.json`.

`load_feds` raises `ValueError` for: no records for the fire, a fire ID in more than one year without `year`, no CRS, missing columns, missing times, empty geometry, invalid geometry with `repair=False`, non-polygon geometry, or a target CRS that is not in metres.

## Strategy comparison

- Clone the same environment, history and pheromone for each strategy.
- Use the same random seeds.
- Use the same planning time.
