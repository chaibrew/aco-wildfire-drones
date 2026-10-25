# Continuous fire from 12-hourly FEDS snapshots

FEDS perimeters come from VIIRS overpasses about 12 hours apart. Drones replan every few minutes, so the simulator needs the fire *between* observations. `fire_sim.spread` builds one with simple fire physics, constrained by both bracketing snapshots. The fire never runs past the last snapshot.

## Model

For each pair of snapshots A_k (time t_k) and A_{k+1} (time t_{k+1}), every cell of the 100 m raster that burns in between gets an arrival time:

```text
T(x) = t_k + pace_k^{-1}( tau_f(x) / (tau_f(x) + tau_b(x)) )
```

- `tau_f(x)`: least travel time from the edge of A_k to x.
- `tau_b(x)`: least travel time from x to the edge of A_{k+1}.

Both are shortest paths (Dijkstra, 16-neighbour stencil) through the newly burned band only, because fire cannot cross ground it already burned. The first and last half-cells are measured to the true polygon edges, so thin bands are not biased.

Edge cost is `length / rate`, with a relative directional rate built from three parts:

| Factor | Formula | Source |
| --- | --- | --- |
| Wind | Elliptical spread, rate `(1 - e) / (1 - e cos psi)` relative to the head. Length-to-breadth ratio from midflame wind (0.4 × 10 m wind), capped at 8. | Anderson (1983) |
| Slope | `1 + 21 tan^2(slope)` uphill, 1 downhill. 21 = 5.275 β^-0.3 with β = 0.01. | Rothermel (1972) |
| Heterogeneity | Seeded log-normal multiplier, σ = 0.3, 500 m Gaussian correlation | — |

`pace_k` maps the fraction of the interval's spread to clock time using cumulative `FFWI^p + 1`. FFWI is the Fosberg Fire Weather Index (Fosberg 1978), calculated hourly from Open-Meteo/ERA5 temperature, humidity and wind at the fire centroid. Wind for the ellipse is the FFWI-weighted vector mean over the interval, so shifting winds produce a rounder fire.

**Guarantees.** On the raster, {T ≤ t_k} equals the cumulative observed A_k for every snapshot. Growth is monotone. Absolute spread rates are never needed, because only ratios enter.

**Other rules:**
- Spot fires (pieces of A_{k+1} unreachable from A_k) ignite just after t_k at their cell nearest the old fire.
- Unburned islands with no route out fill from their edge.
- FEDS labels (`T00`/`T12`) are converted to approximate overpass times: 01:30 / 13:30 local solar time, ±1 h.

**What physics can and cannot change.** The snapshots already record how far the fire went in each direction. A speed-up that is uniform along a path cancels in the ratio. Physics reshapes timing where speed varies along a path (steep then flat terrain, heterogeneity), where the fastest way in and out differ (flanks under wind), and through the weather pacing.

## Ground truth vs what the fleet knows

The ArrivalField is ground truth and is never shown to a planner. `run_mission` keeps the two apart.

**Fleet map (`FireMap`).** This is the planner's only input.
- **At FEDS observations:** the planner learns the observed area (the true burned area at the snapshot time), after an optional latency.
- **From drones:** an observation finishing at (x, y, t) reveals the true burned state within the sensor radius R (default: the cell size) at time t.
- **Unobserved ground:** assumed unchanged since it was last seen. When the fire outruns a footprint, the map only knows "burned up to here", so drones chase the front. Undiscovered spot fires stay invisible until a drone passes or the next snapshot arrives.
- **Monitoring points:** grid cells on the mapped perimeter (whole perimeter, not just the active part).
- **Risk:** taken from the newest spread evidence for the cell.
  - A snapshot gives the distance each newly burned cell lies beyond the previous snapshot, divided by the interval.
  - A drone gives the distance newly seen fire lies beyond the previously known fire, divided by the time since that spot was last observed.
  - Looking and seeing no advance records zero.
  - Tiers: below 2 m/min → 1, 2–10 → 2, above 10 → 3. `q = base / risk`.

**Footprints in planning.** `PlanningState.coverage` lists, for each point, every point within R of it. A visit observes them all, so `evaluate` credits nearby points too.

**Truth score.** The true active front is found with `FrontMonitor`: perimeter cells where fire reaches a neighbour before the next observation.
- A true cell counts as observed whenever it lies within R of any drone visit.
- A satellite snapshot counts as observing everything at its observation time.
- `true_objective` is J over these cells, so fire the fleet has not discovered goes overdue.
- `front_on_map_fraction` is the share of true front cells that are on the map.

**Change pattern (Camp Fire, 24 h, 6 drones, smoke router, 1 km cells).**
- Median 6% of points change per 10-minute replan, driven by drone discoveries.
- 80% change when a snapshot lands.
- `front_on_map_fraction` falls to about 0.8 before an observation, then resets to 1.

## Validation: leave one snapshot out

Hide snapshot k, interpolate the 24 h gap from k−1 to k+1, and compare with the hidden perimeter. Scores cover only cells that burned during the gap.

- **IoU**: growth-weighted overlap between predicted and observed growth (1 = perfect).
- **Frac err**: absolute error in the share of the 24 h growth completed by t_k.
- **Plain**: straight-line distance with uniform pacing (no physics), on the same raster.
- **Physics** column: wind + terrain + FFWI² pacing, noise off.

| Fire (2018) | Area km² | Intervals | IoU physics | IoU plain | Frac err physics | Frac err plain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3885 (Camp) | 618 | 14 | **0.494** | 0.429 | 0.311 | 0.298 |
| 2519 | 193 | 37 | **0.427** | 0.375 | **0.233** | 0.354 |
| 2502 | 463 | 28 | **0.389** | 0.353 | **0.261** | 0.304 |
| 2644 | 50 | 34 | 0.395 | 0.407 | 0.345 | 0.311 |
| 3616 | 34 | 28 | 0.299 | 0.337 | 0.416 | 0.319 |
| 67 | 19 | 7 | 0.095 | 0.132 | 0.482 | 0.482 |

**Findings:**
- Physics helps on the large, wind-driven fires and is slightly worse on small ones.
- Most of the gain comes from weather pacing. Wind shape and slope each change IoU by about ±0.02.
- A pacing exponent of p = 3 helped large fires more but hurt small ones. p = 2 is the default.
- Absolute IoUs of 0.3–0.5 show that half-daily growth is lumpy and hard to recover by any method. The simulator only needs plausible in-between dynamics: the observed snapshots are always honoured.

Reproduce with `fire-sim fire ... --validate`, which writes `validation.csv`.

## Limitations

- Single-point weather per fire, and no fuel map.
- Suppression is not modelled.
- Projected grid north is treated as true north.
- Rates are relative within an interval, so the risk thresholds use interpolated rates, not observed ones.
- Noise seeds give different in-between dynamics, but every seed shares the same observed snapshots.
