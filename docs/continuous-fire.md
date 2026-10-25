# Continuous fire and fleet map

FEDS gives one perimeter approximately each 12 hours. The drones replan each 10 minutes. Thus the simulation must have the fire between observations. This document describes:

1. The ground-truth fire (`spread.py`, `fire.py`, `weather.py`, `terrain.py`).
2. The fleet map that the planner uses (`firemap.py`, `mission.py`).
3. The validation of the ground-truth fire (`validation.py`).

## 1. Ground-truth fire

### Arrival time

Snapshot A_k is at time t_k. Each cell that burns between A_k and A_{k+1} receives an arrival time:

```text
T(x) = t_k + pace_k^{-1}( tau_f(x) / (tau_f(x) + tau_b(x)) )
```

- `tau_f(x)`: the shortest travel time from the edge of A_k to x.
- `tau_b(x)`: the shortest travel time from x to the edge of A_{k+1}.

This is a constrained form of two published methods:

- Minimum travel time (Finney 2002): fire arrival time is the shortest travel time through a directional spread-rate grid. FlamMap uses this method.
- Arrival time between two perimeters (Farguell et al. 2018): the arrival time must agree with both perimeters. Farguell et al. use a numerical optimiser. This project uses a ratio of two travel times. The ratio agrees with both perimeters exactly.

The area is a grid of 100 m cells. Dijkstra's algorithm finds the fastest path for the fire from cell to cell. From a cell, the fire can move to 16 neighbours: 4 sides, 4 corners and 8 knight moves.

- The fire moves only through the cells that burn in this interval. It cannot go back through ground that burned before.
- A perimeter line does not follow the cell edges. Thus the first and the last step of a path use the real distance from the cell centre to the perimeter line, not a full cell.

### Spread rate

The cost of a step is length divided by rate. The rate has three parts:

| Part | Formula | Source |
| --- | --- | --- |
| Wind | Ellipse: `(1 - e) / (1 - e cos psi)` of the head rate. The length-to-breadth ratio comes from the midflame wind (0.4 × 10 m wind). Maximum ratio: 8. | Anderson (1983) |
| Slope | `1 + 21 tan^2(slope)` uphill. 1 downhill. 21 = 5.275 β^-0.3, β = 0.01. | Rothermel (1972) |
| Variation | Random log-normal factor, σ = 0.3, correlation length 500 m. The seed controls it. | — |

### Pacing

`pace_k` changes the fraction of the spread into clock time. It uses the cumulative sum of `FFWI + 1`. FFWI is the Fosberg Fire Weather Index (Fosberg 1978). The simulation calculates FFWI each hour from the temperature, humidity and wind at the fire centre (Open-Meteo, ERA5).

The wind for the ellipse is the mean wind vector in the interval. FFWI gives the weight of each hour. When the wind direction changes, the mean vector decreases and the fire becomes more round.

### Properties

- At each snapshot time t_k, the burned cells are equal to the FEDS perimeter A_k.
- The burned area always increases.
- The model uses only rate ratios. It does not use a fuel model.

### Special cases

- **Spot fire:** a part of A_{k+1} that the fire cannot reach from A_k. It starts immediately after t_k, at its cell nearest to A_k.
- **Island:** an unburned area that has no path to unburned ground. It burns from its edge.
- **Time labels:** FEDS labels are T00 and T12. The simulation changes them to the satellite times, 01:30 and 13:30 local solar time (accuracy ±1 h).

### Limits of the physics

The snapshots record how far the fire moved in each direction. If a speed increase is constant along a path, it has no effect on the ratio. Physics changes the timing only in these conditions:

- The speed changes along the path, for example steep ground and then flat ground.
- The fastest path in and the fastest path out are in different directions, for example on a flank in wind.
- The weather pacing changes the growth speed with time.

## 2. Ground truth and fleet map

The arrival-time fire is the ground truth. The planner never sees it. `run_mission` keeps the two separate.

### Fleet map (`FireMap`)

The fleet map is the only input to the planner.

- **FEDS snapshot:** at the snapshot time, the map receives the true burned area.
- **Drone observation:** when an observation completes at (x, y, t), the map receives the true burned state in the sensor radius R at time t. The default R is the cell size.
- **Unobserved ground:** the map keeps the last observed state. If the fire moves out of the sensor radius, the map shows the fire only to the edge of the radius. The drones must follow the front. The map does not show a spot fire until a drone or the next snapshot finds it.
- **Monitoring points:** all grid cells on the mapped perimeter.

### Risk

Risk comes from the most recent spread evidence for the cell:

- **Snapshot:** the distance of each new burned cell from the previous snapshot, divided by the interval.
- **Drone:** the distance of the new burned area from the previously known fire, divided by the time since the last observation of that area.
- **No advance:** when a drone sees no advance, the rate is 0.

| Spread rate | Risk | Revisit period q |
| --- | --- | --- |
| Less than 2 m/min | 1 | 900 s |
| 2 to 10 m/min | 2 | 450 s |
| More than 10 m/min | 3 | 300 s |

The first snapshot has no previous snapshot. Thus, if the mission starts at the first snapshot, all points have risk 1 until a drone measures the spread.

### Sensor radius in planning

`PlanningState.coverage` gives, for each point, all points in radius R. One visit observes all of them. `evaluate` records an observation for each of them.

### Truth score

`FrontMonitor` finds the true active front: the perimeter cells where the fire reaches a neighbour before the next observation.

- A true cell is observed when it is in radius R of a drone visit.
- A satellite snapshot observes all cells at its observation time.
- `true_objective` is J on these cells. Fire that the fleet did not find becomes overdue.
- `front_on_map_fraction` is the fraction of the true front cells that are on the fleet map.

### Change pattern

Camp Fire, 24 h, 6 drones, 1 km cells, `smoke_router`:

- Each 10-minute replan changes a median of 8% of the points. Drone observations cause these changes.
- A new snapshot changes 79% of the points.
- `front_on_map_fraction` decreases to approximately 0.8 before a snapshot. After the snapshot, it is 1.

## 3. Validation: leave one snapshot out

Remove snapshot k. Interpolate the 24 h interval from k−1 to k+1. Compare the result at t_k with the removed perimeter. Only the cells that burned in the interval are compared.

- **IoU:** overlap of the calculated growth and the observed growth (1 is perfect).
- **Frac err:** error in the fraction of the 24 h growth that is complete at t_k.
- **Physics:** wind, slope and FFWI pacing. No random variation.
- **Plain:** straight-line distance and constant pacing.

| Fire (2018) | Area km² | Intervals | IoU physics | IoU plain | Frac err physics | Frac err plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3885 (Camp) | 618 | 14 | **0.472** | 0.429 | 0.300 | 0.298 |
| 2519 | 193 | 37 | **0.411** | 0.375 | **0.283** | 0.354 |
| 2502 | 463 | 28 | **0.380** | 0.353 | **0.284** | 0.304 |
| 2644 | 50 | 34 | 0.401 | 0.407 | 0.339 | 0.311 |
| 3616 | 34 | 28 | 0.312 | 0.337 | 0.378 | 0.319 |
| 67 | 19 | 7 | 0.099 | 0.132 | 0.489 | 0.482 |

Results:

- Physics is better on the large, wind-driven fires. It is slightly worse on the small fires.
- Most of the improvement comes from the weather pacing. Wind shape and slope each change IoU by approximately 0.02.
- No method recovers the 12-hour growth accurately (IoU 0.3 to 0.5). The simulation needs only a possible fire between snapshots. It always agrees with the snapshots.

To repeat the check, use `fire-sim fire ... --validate`. The result is in `validation.csv`.

## References

- Anderson, H. E. (1983). Predicting wind-driven wild land fire size and shape. USDA Forest Service Research Paper INT-305.
- Farguell Caus, A., Haley, J., Kochanski, A. K., Cortés Fité, A., Mandel, J. (2018). Assimilation of fire perimeters and satellite detections by minimization of the residual in a fire spread model. ICCS 2018. arXiv:1804.03995.
- Finney, M. A. (2002). Fire growth using minimum travel time methods. Canadian Journal of Forest Research, 32(8), 1420–1424.
- Fosberg, M. A. (1978). Weather in wildland fire management: the fire weather index. Proceedings of the Conference on Sierra Nevada Meteorology.
- Rothermel, R. C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.

## Limits

- Weather comes from one point for each fire. There is no fuel map.
- The model has no fire suppression.
- Grid north is used as true north.
- The truth score uses interpolated spread rates. The fleet map uses observed spread rates.
- Different seeds give different fires between snapshots. All seeds agree with the same snapshots.
