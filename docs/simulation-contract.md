# Simulation contract and modelling decisions

These choices make the starter implementation precise. The group should agree on them before fixing experimental parameters.

## Data flow

```text
FEDS GeoPackage (perimeter layer; year + fireID)
  -> sorted polygon snapshots in one projected metre CRS
  -> fixed grid / active points / change sets
  -> PlanningState -> ACO, greedy or VNS -> routes
  -> pure candidate evaluation -> execute chosen routes
  -> carry positions + observation history into next stage
```

`load_feds` returns a list of `Perimeter(timestamp, geometry)` and provenance. It retains polygon components and holes. Duplicate selected records at a timestamp are unioned, with the count recorded. Missing IDs, missing CRS, nonpolygon geometry, missing times and invalid geometry produce errors. Multi-year datasets require a year when a fire ID repeats. A single tracked FEDS ID is used; the adapter does not stitch IDs across fire mergers. Inspect heritage data separately if the research needs a physical incident spanning multiple IDs.

## Grid, priorities and changes

`Grid` uses a fixed world origin `(0, 0)` and fixed cell size for the entire sequence. IDs are integer lattice coordinates, never array positions. Cells intersecting the polygon boundary become active; fully interior burned cells do not. Internal hole boundaries are included. Closed-cell intersection includes both adjacent cells for a boundary exactly on a grid line, including corner-only touches. This deliberate convention is covered by tests.

Each monitoring location is the cell centre, so the target is an approximation to the perimeter and may lie on either side of it. Maximum centre-to-boundary offset is half the cell diagonal. Exact sensor footprints, collision avoidance, no-fly zones and return-to-base constraints are outside this implementation.

Synthetic risk is a deterministic hash of `(seed, cell_id)`, taking values 1, 2 or 3. `q_i = base_revisit_s / risk_i`. It is independent of routing strategy and stable across snapshots. This is not inferred hazard, fire intensity or a validated risk model. Planners may use a different priority assignment if all variants receive the same assignment.

`perimeter_changes` computes:

- `added`: current IDs absent previously.
- `removed`: previous IDs absent currently.
- `changed`: surviving IDs whose attributes or within-cell boundary geometry changed.
- `affected_current = added | changed`: the current `C_t` for selective reset.

Geometry comparison is topological equality after clipping the boundary to each surviving cell. Very small geometric differences can flag a cell; there is no displacement tolerance. State that rule in experiment methods or add a pilot-selected tolerance before testing.

Reported magnitude is `|added ∪ removed ∪ changed| / |previous ∪ current|`. Initial activation reports fraction 1 and is **not** an adaptation event. For changes containing only deletions, `C_t` can be empty; deleted vertices and their incident edges still need to be removed.

## Planner contract

`PlanningState` provides simulation time, planning horizon, speed, service duration, drone positions, monitoring points, and `last_seen` times. Points and history are copied when making a state. Treat the state as read-only. Distances are Euclidean metres and travel duration is distance/speed. Depot-to-point travel must use each drone's current position.

Use `sorted(state.points)` to build deterministic dense matrix indices. Save that ID-to-index mapping for that stage only. On a change, remap pheromone by stable IDs; copying matrix rows by array index would transfer pheromone to unrelated points. New edges start at `tau_0`, deleted edges disappear, and surviving edges retain their values unless the chosen strategy resets them. Diagonal pheromone and repeated same-point visits need an explicit policy in the ACO implementation.

`coverage` (optional) maps each point to the points observed by visiting it: every point within the sensor radius, including itself. When it is `None`, a visit observes only the visited point. The returned `Evaluation.visits` lists `(time, drone_id, point_id)` for every completed observation.

The planner returns exactly one ordered ID list per drone, including empty lists. Repeated IDs and shared visits are allowed. Every leg, including service at the destination, must finish within `H`. No mandatory return is imposed. A drone finishes at its last visited point and idles for the remaining time; it does not automatically observe while idling. Each explicit visit completes after a strictly positive service duration, including repeated same-point visits. This prevents infinite instantaneous visits. Plans extending beyond `H` are rejected atomically rather than silently truncated. Mid-flight replanning is therefore not modelled.

For strategy comparisons, clone the same starting environment/history and pre-change pheromone into each strategy, use paired seeds, and impose the same planning budget. The starter smoke CLI runs one deterministic plumbing router and does not implement this experiment harness or adaptation convergence traces.

## Revisit accounting

Observations occur at service completion. Observations by different drones are merged and sorted per point; simultaneous events count once.

For a point active during `[t, t+H]`, build the times:

```text
[last observation (or activation), observations in this stage, t+H]
```

`P_i` is the largest consecutive difference, and `J = max_i max(0, P_i/q_i - 1)`. This includes the terminal censored interval, so an unvisited point is not accidentally scored as zero. A gap may start before the stage: overdue history carries across replans. This is the implementation's explicit extension of the proposal's finite-horizon objective and should be stated in the report.

New points have initial age zero at activation, without counting activation as a real observation. Surviving points preserve history. Removed points leave the objective and their history is discarded; a reappearing cell begins a new active lifetime. If the group chooses to preserve history across inactive periods, change this policy and its tests together.

An empty active set has objective, counts and distances zero. `overdue_count` counts points with `P_i > q_i` during the stage, not just points overdue at the final instant. Mean/max gap metrics aggregate the `P_i` values, not every inter-observation interval. Route cost includes travel from the current position, excludes service time, and has no implicit depot return.

## Time and experiment limitations

The CLI holds the current source geometry until the next source timestamp, divided by `time_scale`. It never interpolates geometry or predicts spread. Source time labels and simulation seconds are both saved. Revisit periods and speed use simulation units and do not automatically rescale when `time_scale` changes.

The last frame has no known successor and is held for the explicitly configured final horizon. A static run repeats the same source geometry/time label while simulation time advances. A controlled perturbation is a synthetic addition even when its starting geometry came from FEDS.

Tests cover metric projection, actual GeoPackage round trips, chronological ordering, ambiguous fire IDs, geometry repair, grid identity, boundary selection, change detection, route feasibility, repeated/simultaneous observations, unvisited points, carryover, removal/reappearance, empty environments and deterministic smoke routes. Static ACO validation, exhaustive optimality checks, VNS validation and statistical experiment design belong to the algorithms/harness work that follows.
