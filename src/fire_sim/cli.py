import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from shapely.geometry import mapping

from .feds import inspect_file, load_feds
from .grid import Grid, perimeter_changes
from .models import Perimeter
from .scenarios import perturb, synthetic_sequence
from .simulation import Drone, Environment, smoke_router


def preview(sequence, point_sets, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for ax, index in zip(axes[0], [0, len(sequence) - 1]):
        shape = sequence[index].geometry
        parts = list(shape.geoms) if shape.geom_type == "MultiPolygon" else [shape]
        for p in parts:
            x, y = p.exterior.xy
            ax.fill([v / 1000 for v in x], [v / 1000 for v in y], color="#f5b48a", alpha=0.55)
            for ring in p.interiors:
                x, y = ring.xy
                ax.fill([v / 1000 for v in x], [v / 1000 for v in y], color="white")
        pts = list(point_sets[index].values())
        scatter = ax.scatter([p.x / 1000 for p in pts], [p.y / 1000 for p in pts],
                             c=[p.risk for p in pts], s=13, cmap="viridis", vmin=1, vmax=3)
        ax.set(title=f"Stage {index}: {len(pts)} monitoring points", xlabel="Projected x (km)", ylabel="Projected y (km)")
        ax.set_aspect("equal")
    fig.colorbar(scatter, ax=list(axes[0]), ticks=[1, 2, 3], label="Synthetic risk tier", shrink=0.7)
    all_bounds = [p.geometry.bounds for p in sequence]
    xmin, ymin = min(b[0] for b in all_bounds), min(b[1] for b in all_bounds)
    xmax, ymax = max(b[2] for b in all_bounds), max(b[3] for b in all_bounds)
    margin = max(xmax - xmin, ymax - ymin) * .08 + 1
    for ax in axes[0]:
        ax.set_xlim((xmin - margin) / 1000, (xmax + margin) / 1000)
        ax.set_ylim((ymin - margin) / 1000, (ymax + margin) / 1000)
    minutes = [r["simulation_start_s"] / 60 for r in rows]
    axes[1, 0].plot(minutes, [r["active_points"] for r in rows], "o-", label="Active points")
    axes[1, 0].plot(minutes, [r["affected_current"] for r in rows], "o-", label="Affected current points")
    axes[1, 0].set(xlabel="Simulation time (minutes)", ylabel="Count")
    axes[1, 0].legend()
    axes[1, 1].plot(minutes, [r["objective"] for r in rows], "o-", color="#b44627")
    axes[1, 1].set(xlabel="Simulation time (minutes)", ylabel="Worst normalised overdue time")
    fig.suptitle("Perimeter replay • smoke router • synthetic priorities", fontsize=15)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def run(args):
    if args.steps < 1 or args.drones < 1:
        raise ValueError("steps and drones must be positive")
    if not all(math.isfinite(v) and v > 0 for v in (args.horizon, args.time_scale)):
        raise ValueError("horizon and time-scale must be finite and positive")
    if args.command == "feds":
        sequence, provenance = load_feds(args.path, fire_id=args.fire_id, year=args.year,
                                         target_crs=args.crs, repair=args.repair, limit=args.steps)
    else:
        sequence = synthetic_sequence(args.steps)
        provenance = {"source": "synthetic (not FEDS)", "target_crs": "local Cartesian metres"}
    if args.static:
        sequence = [sequence[0]] * args.steps
    if args.perturb_stage is not None:
        if not 1 <= args.perturb_stage < len(sequence):
            raise ValueError("perturb-stage must lie between 1 and the last stage")
        sequence = [Perimeter(p.timestamp, perturb(p.geometry, locality=args.locality,
                     radius_m=args.radius, patches=args.patches, seed=args.seed))
                    if k >= args.perturb_stage else p for k, p in enumerate(sequence)]
    grid = Grid(cell_size_m=args.cell_size, base_revisit_s=args.revisit, seed=args.seed)
    xmin, ymin, _, _ = sequence[0].geometry.bounds
    depot = (xmin - args.cell_size, ymin - args.cell_size)
    env = Environment([Drone(i, *depot) for i in range(args.drones)],
                      speed_mps=args.speed, service_s=args.service)
    rows, snapshots, point_sets = [], [], []
    previous, previous_geometry = {}, sequence[0].geometry
    for k, perimeter in enumerate(sequence):
        points = grid.points(perimeter.geometry)
        delta = perimeter_changes(grid, previous_geometry, perimeter.geometry, previous, points)
        horizon = args.horizon
        if not args.static and k + 1 < len(sequence):
            horizon = (sequence[k + 1].timestamp - perimeter.timestamp).total_seconds() / args.time_scale
        env.activate(points)
        state = env.state(horizon)
        started = time.perf_counter()
        routes = smoke_router(state)
        planning_s = time.perf_counter() - started
        result = env.step(routes, horizon)
        rows.append({"stage": k, "source_timestamp": perimeter.timestamp.isoformat(),
                     "simulation_start_s": state.time_s, "horizon_s": horizon,
                     "active_points": len(points), "added": len(delta.added),
                     "removed": len(delta.removed), "changed": len(delta.changed),
                     "affected_current": len(delta.affected_current),
                     "affected_union_fraction": delta.fraction(previous, points),
                     "planning_s": planning_s, **result.summary()})
        snapshots.append({"stage": k, "source_timestamp": perimeter.timestamp.isoformat(),
                          "simulation_start_s": state.time_s, "horizon_s": horizon,
                          "perimeter": mapping(perimeter.geometry),
                          "points": [{**asdict(p), "last_seen_s": state.last_seen[i]} for i, p in points.items()],
                          "drones": [asdict(d) for d in state.drones],
                          "changes": {"added": sorted(delta.added), "removed": sorted(delta.removed),
                                      "changed": sorted(delta.changed), "affected_current": sorted(delta.affected_current)},
                          "routes": {str(i): r for i, r in routes.items()},
                          "observations": [{"id": i, "times_s": visits} for i, visits in result.observations.items()]})
        point_sets.append(points)
        previous, previous_geometry = points, perimeter.geometry
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    config = {k: v for k, v in vars(args).items()}
    payload = {"schema_version": 1, "units": {"distance": "metres", "simulation_time": "seconds"},
               "provenance": provenance, "config": config,
               "priority_model": "synthetic deterministic cell hash; q = base_revisit / risk; risk in {1,2,3}",
               "router": "smoke_router (not a validated experimental baseline)", "snapshots": snapshots}
    (out / "scenario.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    preview(sequence, point_sets, rows, out / "preview.png")
    print(json.dumps({"output": str(out.resolve()), "stages": len(rows),
                      "point_counts": [r["active_points"] for r in rows],
                      "last_stage_metrics": rows[-1]}, indent=2))


def _fire_options(p):
    p.add_argument("path")
    p.add_argument("--fire-id", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--limit", type=int, help="Use only the first N snapshots")
    p.add_argument("--crs", help="Default: UTM estimated from first perimeter")
    p.add_argument("--res", type=float, default=100.0, help="Fine raster cell size (m)")
    p.add_argument("--seed", type=int, default=0, help="Spread heterogeneity seed")
    p.add_argument("--noise", type=float, default=0.3, help="Log-normal heterogeneity sigma (0 = off)")
    p.add_argument("--pacing-exponent", type=float, default=2.0, help="Growth weight = FFWI**p")
    p.add_argument("--no-weather", action="store_true", help="Uniform pacing, no wind")
    p.add_argument("--no-terrain", action="store_true", help="Flat ground")
    p.add_argument("--time-convention", choices=["overpass", "label"], default="overpass")


def _load_field(args):
    from .fire import load_fire
    from .spread import SpreadModel
    return load_fire(args.path, fire_id=args.fire_id, year=args.year, limit=args.limit, crs=args.crs,
                     res=args.res, seed=args.seed, model=SpreadModel(noise_sigma=args.noise),
                     weather=not args.no_weather, terrain=not args.no_terrain,
                     pacing_exponent=args.pacing_exponent, convention=args.time_convention)


def _monitor(field, args):
    from .monitor import FrontMonitor
    return FrontMonitor(field, cell_size_m=args.cell_size, base_revisit_s=args.revisit,
                        rate_tiers_mpm=tuple(args.rate_tiers), active_only=not args.whole_perimeter,
                        active_window_s=None if args.active_window_h is None else args.active_window_h * 3600)


def fire_command(args):
    from . import plots
    from .validation import leave_one_out, summarise
    field, sequence = _load_field(args)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    field.save(out / "field")
    monitor = _monitor(field, args)
    plots.plot_fire(field, sequence, monitor, out / "fire.png")
    if args.gif:
        plots.animate(field, monitor, out / "fire.gif", step_s=args.gif_step_h * 3600)
    report = {"output": str(out.resolve()), "raster": field.raster.shape, "snapshots": len(field.snapshot_s),
              "hours": field.end_s / 3600, "intervals": field.meta["intervals"]}
    if args.validate:
        from .spread import SpreadModel
        rows = leave_one_out(sequence, field.crs, res=args.res, model=SpreadModel(noise_sigma=0.0),
                             weather=not args.no_weather, terrain=not args.no_terrain,
                             pacing_exponent=args.pacing_exponent, convention=args.time_convention)
        if rows:
            with (out / "validation.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        report["validation"] = summarise(rows)
    print(json.dumps(report, indent=2, default=str))


def mission_command(args):
    from .mission import MissionConfig, run_mission
    from .plots import animate_mission, plot_mission
    field, _ = _load_field(args)
    config = MissionConfig(cell_size_m=args.cell_size, sensor_radius_m=args.sensor_radius,
                           base_revisit_s=args.revisit, rate_tiers_mpm=tuple(args.rate_tiers),
                           snapshot_latency_s=args.snapshot_latency_h * 3600, drones=args.drones,
                           speed_mps=args.speed, service_s=args.service, horizon_s=args.horizon)
    start = field.start_s + args.start_h * 3600
    stop = None if args.duration_h is None else start + args.duration_h * 3600
    frames = []

    def record(stage, fire_map, points, routes):
        if args.gif and stage % args.gif_every == 0:
            frames.append((stage, fire_map.known.copy(), dict(points), routes))

    rows = run_mission(field, smoke_router, config, start_s=start, stop_s=stop, record=record)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "config.json").write_text(json.dumps({"args": vars(args), "mission": asdict(config),
                                                 "fire": field.meta}, indent=2, default=str) + "\n")
    plot_mission(field, rows, out / "mission.png")
    if args.gif:
        animate_mission(field, rows, frames, out / "mission.gif")
    fractions = [r["affected_union_fraction"] for r in rows[1:]]
    print(json.dumps({"output": str(out.resolve()), "stages": len(rows),
                      "map_points_median": sorted(r["map_points"] for r in rows)[len(rows) // 2],
                      "changed_fraction_median": sorted(fractions)[len(fractions) // 2] if fractions else None,
                      "worst_true_objective": max(r["true_objective"] for r in rows),
                      "mean_front_on_map": sum(r["front_on_map_fraction"] for r in rows) / len(rows)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="List source layers and year/fire IDs")
    inspect.add_argument("path")
    for name in ("demo", "feds"):
        p = sub.add_parser(name)
        if name == "feds":
            p.add_argument("path")
            p.add_argument("--fire-id", required=True)
            p.add_argument("--year", type=int)
            p.add_argument("--crs", help="Default: UTM estimated from first perimeter")
            p.add_argument("--repair", action="store_true")
        p.add_argument("--output", default=f"outputs/{name}")
        p.add_argument("--steps", type=int, default=6)
        p.add_argument("--static", action="store_true", help="Freeze first perimeter for all stages")
        p.add_argument("--cell-size", type=float, default=500.0, help="Metres")
        p.add_argument("--drones", type=int, default=3)
        p.add_argument("--speed", type=float, default=15.0, help="Metres per second")
        p.add_argument("--service", type=float, default=5.0, help="Seconds per observation")
        p.add_argument("--revisit", type=float, default=900.0, help="Base q in simulation seconds")
        p.add_argument("--horizon", type=float, default=600.0, help="Final/static stage seconds")
        p.add_argument("--time-scale", type=float, default=72.0, help="Source seconds per simulation second")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--perturb-stage", type=int, help="Apply geometric growth from this zero-based stage onwards")
        p.add_argument("--locality", choices=["local", "distributed"], default="local")
        p.add_argument("--radius", type=float, default=400.0, help="Growth patch radius in metres")
        p.add_argument("--patches", type=int, default=4)
    for name in ("fire", "mission"):
        p = sub.add_parser(name, help={"fire": "Build a continuous fire from FEDS + weather + terrain",
                                       "mission": "Fly drones that plan on their own map of a hidden continuous fire"}[name])
        _fire_options(p)
        p.add_argument("--output", default=f"outputs/{name}")
        p.add_argument("--cell-size", type=float, default=500.0, help="Monitoring grid cell (m)")
        p.add_argument("--revisit", type=float, default=900.0, help="Base q (s); q = base / risk")
        p.add_argument("--rate-tiers", type=float, nargs=2, default=[2.0, 10.0], metavar=("LOW", "HIGH"),
                       help="Front spread rate (m/min) for risk 2 and risk 3")
        p.add_argument("--whole-perimeter", action="store_true", help="Monitor every perimeter cell, not just the active front")
        p.add_argument("--active-window-h", type=float, help="Active if fire reaches a neighbour within this many hours (default: before next observation)")
        if name == "fire":
            p.add_argument("--validate", action="store_true", help="Leave-one-snapshot-out check")
            p.add_argument("--gif", action="store_true")
            p.add_argument("--gif-step-h", type=float, default=0.5)
        else:
            p.add_argument("--start-h", type=float, default=0.0, help="Hours after first snapshot")
            p.add_argument("--duration-h", type=float, help="Default: until last snapshot")
            p.add_argument("--horizon", type=float, default=600.0, help="Replanning interval (s)")
            p.add_argument("--drones", type=int, default=3)
            p.add_argument("--speed", type=float, default=15.0, help="Metres per second")
            p.add_argument("--service", type=float, default=5.0, help="Seconds per observation")
            p.add_argument("--sensor-radius", type=float, help="Drone footprint radius (m); default = cell size")
            p.add_argument("--snapshot-latency-h", type=float, default=0.0,
                           help="Hours before a FEDS snapshot reaches the planner")
            p.add_argument("--gif", action="store_true")
            p.add_argument("--gif-every", type=int, default=3, help="Stages per animation frame")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_file(args.path), indent=2, default=str))
        elif args.command == "fire":
            fire_command(args)
        elif args.command == "mission":
            mission_command(args)
        else:
            run(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
