# Simulation contract

These rules apply to all planners. The group must agree on them before the experiments.

## Data flow

```text
FEDS GeoPackage
  -> perimeters in one metric CRS (load_feds)
  -> ground-truth arrival-time fire (ArrivalField)
  -> fleet map from snapshots and drone observations (FireMap)
  -> PlanningState -> planner -> routes
  -> evaluate -> Environment.step
  -> drone observations update the fleet map
```

## FEDS loader

`load_feds` returns a list of `Perimeter(timestamp, geometry)` and provenance data.

- It keeps polygon parts and holes.
- It joins duplicate records at one timestamp and records the number.
- A missing ID, missing CRS, non-polygon geometry, missing time or invalid geometry causes an error. Use `--repair` to repair invalid geometry. The provenance records the number of repairs.
- If a fire ID occurs in more than one year, you must give the year.
- It uses one FEDS fire ID. It does not join IDs when fires merge.
- It selects the projection from the first perimeter only.

## Grid and point IDs

- The grid has a fixed origin (0, 0) and a fixed cell size. A cell ID is `(grid_x, grid_y)`. It is not an array position.
- The monitoring point is the cell centre. The maximum distance from the centre to the perimeter is half of the cell diagonal.
- The model has no collision avoidance, no no-fly zones and no return to base.

## Changes

`changes(previous, current)` gives:

- `added`: current IDs that were not in the previous set.
- `removed`: previous IDs that are not in the current set.
- `changed`: IDs in both sets with a different risk or revisit period.
- `affected_current = added | changed`: C_t for selective reset.

`fraction` is `|added ∪ removed ∪ changed| / |previous ∪ current|`. The first activation has fraction 1. It is not an adaptation event. If a change has only removed points, C_t is empty. You must still delete the edges of the removed points.

The legacy commands use `perimeter_changes`. It also marks a cell as changed when the boundary in the cell changes.

## Planner contract

`PlanningState` contains: time, horizon, speed, observation time, drone positions, points, `last_seen` and `coverage`. Do not change the state.

- Distances are Euclidean, in metres. Travel time is distance divided by speed.
- The first leg starts at the current position of the drone.
- Use `sorted(state.points)` to make matrix indices. The indices are correct for one stage only.
- When the points change, move pheromone by cell ID, not by matrix index. New edges start at `tau_0`. Delete the edges of removed points.
- `coverage` gives, for each point, all points in the sensor radius (with the point). If `coverage` is `None`, a visit observes only the visited point.

The planner returns one ordered list of IDs for each drone. A list can be empty.

- A route can visit a point more than one time. Drones can visit the same point.
- All legs and observations must end in the horizon H. `evaluate` rejects a longer route. It does not cut the route.
- A drone stops at its last point. It does not observe while it waits.
- There is no replan during a flight.

`evaluate(state, routes)` does not change the environment. Use it to compare candidates. Use `env.step(routes, H)` to fly the selected routes. `Evaluation.visits` lists `(time, drone_id, point_id)` for each observation.

For strategy comparisons:

- Start each strategy from the same environment, history and pheromone.
- Use the same random seeds.
- Use the same planning time.

## Revisit accounting

An observation occurs when the drone completes its observation time. If two observations occur at the same time, they count as one.

For a point in stage [t, t+H], make this list of times:

```text
[last observation (or activation), observations in this stage, t+H]
```

`P_i` is the largest difference between two adjacent times. `J = max_i max(0, P_i / q_i - 1)`.

- The last interval (to t+H) is included. Thus an unvisited point has a gap.
- A gap can start before the stage. The overdue time continues across replans. This is an extension of the proposal objective. State it in the report.
- A new point starts at age 0. Activation is not an observation.
- A removed point leaves the objective. If it comes back, it starts a new life.
- With no points, J, the counts and the distances are 0.
- `overdue_count` counts the points with `P_i > q_i` in the stage.
- Route distance starts at the current position. It does not contain observation time or a return to base.

## Legacy commands

`fire-sim demo` and `fire-sim feds` keep each FEDS perimeter constant until the next one. They divide time by `--time-scale` (default 72). Speed and revisit periods do not change with the time scale. Risk is a fixed hash of `(seed, cell_id)`. These commands do not use the fleet map.
