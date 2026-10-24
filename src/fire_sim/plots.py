"""Figures for continuous fires: arrival-time map, growth curve, active front, animation."""

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _outline(ax, geometry, **style):
    parts = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    for p in parts:
        for ring in (p.exterior, *p.interiors):
            x, y = ring.xy
            ax.plot(np.asarray(x) / 1000, np.asarray(y) / 1000, **style)


def _extent(raster):
    x0, y0, x1, y1 = raster.bounds
    return [x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000]


def _points(ax, points, size=14):
    pts = list(points.values())
    if pts:
        return ax.scatter([p.x / 1000 for p in pts], [p.y / 1000 for p in pts], c=[p.risk for p in pts],
                          cmap="plasma", vmin=1, vmax=3, s=size, edgecolors="black", linewidths=0.3, zorder=3)


def plot_fire(field, sequence, monitor, output):
    """Arrival-time map with FEDS outlines, growth curve, active front, active point count."""
    plt = _plt()
    hours = np.where(np.isfinite(field.arrival_s), field.arrival_s / 3600, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    ax = axes[0, 0]
    image = ax.imshow(np.ma.masked_invalid(hours), origin="lower", extent=_extent(field.raster), cmap="turbo")
    for p in sequence:
        _outline(ax, p.geometry, color="black", linewidth=0.5)
    fig.colorbar(image, ax=ax, label="Interpolated arrival (hours after first snapshot)", shrink=0.8)
    ax.set(title="Arrival time; black = observed FEDS perimeters", xlabel="x (km)", ylabel="y (km)", aspect="equal")

    times = np.linspace(field.start_s, field.end_s, 400)
    cell_km2 = (field.raster.res / 1000) ** 2
    area = [np.count_nonzero(field.arrival_s <= t) * cell_km2 for t in times]
    observed = [np.count_nonzero(field.arrival_s <= s) * cell_km2 for s in field.snapshot_s]
    ax = axes[0, 1]
    ax.plot(times / 3600, area, label="Continuous fire")
    ax.plot(np.array(field.snapshot_s) / 3600, observed, "o", color="black", label="FEDS snapshots")
    ax.set(title="Burned area", xlabel="Hours after first snapshot", ylabel="km²")
    ax.legend()

    t_show = field.snapshot_s[min(1, len(field.snapshot_s) - 1)] / 2
    ax = axes[1, 0]
    ax.imshow(np.ma.masked_where(field.arrival_s > t_show, np.ones(field.raster.shape)), origin="lower",
              extent=_extent(field.raster), cmap="Oranges", vmin=0, vmax=1.6, alpha=0.6)
    scatter = _points(ax, monitor.points(t_show))
    if scatter is not None:
        fig.colorbar(scatter, ax=ax, ticks=[1, 2, 3], label="Risk tier (from spread rate)", shrink=0.8)
    ax.set(title=f"Active-front monitoring points at t = {t_show / 3600:.1f} h",
           xlabel="x (km)", ylabel="y (km)", aspect="equal")

    ax = axes[1, 1]
    sample = np.arange(field.start_s, field.end_s, 1800)
    ax.plot(sample / 3600, [len(monitor.points(t)) for t in sample])
    for s in field.snapshot_s:
        ax.axvline(s / 3600, color="grey", linewidth=0.5, linestyle=":")
    ax.set(title="Active monitoring points (dotted = FEDS observation times)",
           xlabel="Hours after first snapshot", ylabel="Points")
    fig.suptitle(f"Continuous fire from FEDS fire {field.meta.get('provenance', {}).get('fire_id', '?')}: "
                 f"{field.meta.get('weather', {}).get('source', '')[:40]}; terrain: {field.meta.get('terrain')}",
                 fontsize=12)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def animate(field, monitor, output, step_s=1800, max_frames=120, fps=8):
    """GIF of the burning area and active monitoring points."""
    plt = _plt()
    from matplotlib.animation import FuncAnimation, PillowWriter
    times = np.arange(field.start_s, field.end_s + 1, step_s)[:max_frames]
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    extent = _extent(field.raster)

    def frame(t):
        ax.clear()
        burned = field.arrival_s <= t
        recent = burned & (field.arrival_s > t - 3 * 3600) & ~field.initial
        ax.imshow(np.where(recent, 2, np.where(burned, 1, np.nan)), origin="lower", extent=extent,
                  cmap="YlOrRd", vmin=0, vmax=2.5, interpolation="nearest")
        _points(ax, monitor.points(t), size=18)
        ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="x (km)", ylabel="y (km)",
               title=f"t = {t / 3600:5.1f} h   (dark = burned in last 3 h; dots = active points)")

    animation = FuncAnimation(fig, frame, frames=times)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_replay(rows, output):
    plt = _plt()
    hours = np.array([r["time_s"] for r in rows]) / 3600
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, layout="constrained")
    axes[0].plot(hours, [r["active_points"] for r in rows], label="active")
    axes[0].plot(hours, [r["affected_current"] for r in rows], label="affected C_t")
    axes[0].set(ylabel="Points")
    axes[0].legend()
    axes[1].plot(hours, [r["affected_union_fraction"] for r in rows], color="#b44627")
    axes[1].set(ylabel="Changed fraction")
    axes[2].plot(hours, [r["objective"] for r in rows], color="black")
    axes[2].set(ylabel="J_overdue", xlabel="Hours after first snapshot")
    fig.suptitle("Replay with smoke_router (plumbing check, not a baseline)")
    fig.savefig(output, dpi=140)
    plt.close(fig)
