import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .feds import inspect_file
from .simulation import smoke_router


def _fire_options(p):
    p.add_argument("path")
    p.add_argument("--fire-id", required=True)
    p.add_argument("--year", type=int)
    p.add_argument("--limit", type=int, help="Use only the first N snapshots")
    p.add_argument("--crs", help="Default: UTM estimated from first perimeter")
    p.add_argument("--res", type=float, default=100.0, help="Fine raster cell size (m)")
    p.add_argument("--seed", type=int, default=0, help="Spread heterogeneity seed")
    p.add_argument("--noise", type=float, default=0.3, help="Log-normal heterogeneity sigma (0 = off)")
    p.add_argument("--no-weather", action="store_true", help="Uniform pacing, no wind")
    p.add_argument("--no-terrain", action="store_true", help="Flat ground")
    p.add_argument("--time-convention", choices=["overpass", "label"], default="overpass")


def _load_field(args):
    from .fire import load_fire
    from .spread import SpreadModel
    return load_fire(args.path, fire_id=args.fire_id, year=args.year, limit=args.limit, crs=args.crs,
                     res=args.res, seed=args.seed, model=SpreadModel(noise_sigma=args.noise),
                     weather=not args.no_weather, terrain=not args.no_terrain, convention=args.time_convention)


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
                             convention=args.time_convention)
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
                           drones=args.drones,
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
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
