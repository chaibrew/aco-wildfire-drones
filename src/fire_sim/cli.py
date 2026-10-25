import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from .config import load_config
from .feds import inspect_file
from .simulation import smoke_router


def _load_field(config):
    from .fire import load_fire
    from .spread import SpreadModel
    d, f = config.data, config.fire
    return load_fire(d.path, fire_id=d.fire_id, year=d.year, limit=d.limit, crs=d.crs,
                     res=f.res_m, seed=f.seed, model=SpreadModel(noise_sigma=f.noise_sigma),
                     weather=f.weather, terrain=f.terrain, convention=f.time_convention)


def _output_dir(config, name):
    out = Path(name)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.source, out / "config.toml")
    return out


def _write_csv(rows, path):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fire_command(config):
    from . import plots
    from .monitor import FrontMonitor
    from .validation import leave_one_out, summarise
    field, sequence = _load_field(config)
    out = _output_dir(config, config.output.fire_dir)
    field.save(out / "field")
    m = config.mission
    monitor = FrontMonitor(field, cell_size_m=m.cell_size_m, base_revisit_s=m.base_revisit_s,
                           rate_tiers_mpm=m.rate_tiers_mpm)
    plots.plot_fire(field, sequence, monitor, out / "fire.png")
    if config.output.gif:
        plots.animate(field, monitor, out / "fire.gif", step_s=config.output.gif_step_h * 3600)
    report = {"output": str(out.resolve()), "raster": field.raster.shape, "snapshots": len(field.snapshot_s),
              "hours": field.end_s / 3600, "intervals": field.meta["intervals"]}
    if config.output.validate:
        from .spread import SpreadModel
        f = config.fire
        rows = leave_one_out(sequence, field.crs, res=f.res_m, model=SpreadModel(noise_sigma=0.0),
                             weather=f.weather, terrain=f.terrain, convention=f.time_convention)
        if rows:
            _write_csv(rows, out / "validation.csv")
        report["validation"] = summarise(rows)
    print(json.dumps(report, indent=2, default=str))


def mission_command(config):
    from .mission import run_mission
    from .plots import animate_mission, plot_mission
    field, _ = _load_field(config)
    start = field.start_s + config.run.start_h * 3600
    stop = None if config.run.duration_h is None else start + config.run.duration_h * 3600
    frames = []

    def record(stage, fire_map, points, routes):
        if config.output.gif and stage % config.output.gif_every == 0:
            frames.append((stage, fire_map.known.copy(), dict(points), routes))

    rows = run_mission(field, smoke_router, config.mission, start_s=start, stop_s=stop, record=record)
    out = _output_dir(config, config.output.mission_dir)
    _write_csv(rows, out / "metrics.csv")
    (out / "run.json").write_text(json.dumps({"config": asdict(config), "fire": field.meta},
                                             indent=2, default=str) + "\n")
    plot_mission(field, rows, out / "mission.png")
    if config.output.gif:
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
    sub.add_parser("inspect", help="List source layers and year/fire IDs").add_argument("path")
    sub.add_parser("fire", help="Build the ground-truth fire").add_argument("config", help="TOML config file")
    sub.add_parser("mission", help="Fly a drone mission").add_argument("config", help="TOML config file")
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_file(args.path), indent=2, default=str))
        elif args.command == "fire":
            fire_command(load_config(args.config))
        else:
            mission_command(load_config(args.config))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
