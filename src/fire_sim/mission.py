import math
import time
from dataclasses import dataclass

import numpy as np

from .firemap import FireMap
from .grid import changes
from .monitor import FrontMonitor
from .simulation import Drone, Environment, gap_metrics


@dataclass
class MissionConfig:
    cell_size_m: float = 500.0
    sensor_radius_m: float | None = None
    base_revisit_s: float = 900.0
    rate_tiers_mpm: tuple[float, float] = (2.0, 10.0)
    snapshot_latency_s: float = 0.0
    drones: int = 3
    speed_mps: float = 15.0
    service_s: float = 5.0
    horizon_s: float = 600.0

    @property
    def radius(self):
        return self.cell_size_m if self.sensor_radius_m is None else self.sensor_radius_m


class TruthTracker:
    def __init__(self, monitor, radius):
        self.monitor = monitor
        self.radius = radius
        self.last_seen = {}

    def score(self, t, end, visits, positions, satellite=()):
        points = self.monitor.points(t)
        self.last_seen = {i: self.last_seen.get(i, t) for i in points}
        observations = {i: [s for s in satellite if t <= s <= end] for i in points}
        if points and visits:
            ids = list(points)
            xy = np.array([(points[i].x, points[i].y) for i in ids])
            seen = np.array([positions[v[2]] for v in visits])
            near = np.hypot(*(xy[:, None, :] - seen[None, :, :]).transpose(2, 0, 1)) <= self.radius + 1e-6
            for row, i in enumerate(ids):
                observations[i].extend(visits[k][0] for k in np.flatnonzero(near[row]))
        result = gap_metrics(points, self.last_seen, observations, end)
        for i, times in observations.items():
            if times:
                self.last_seen[i] = max(self.last_seen[i], max(times))
        return points, result


def run_mission(field, router, config=MissionConfig(), *, start_s=None, stop_s=None, record=None):
    start = field.start_s if start_s is None else start_s
    stop = field.end_s if stop_s is None else min(stop_s, field.end_s)
    if not field.start_s <= start < stop:
        raise ValueError("start must lie inside the observed span")
    latency = config.snapshot_latency_s
    fire_map = FireMap(field, cell_size_m=config.cell_size_m, sensor_radius_m=config.radius,
                       base_revisit_s=config.base_revisit_s, rate_tiers_mpm=config.rate_tiers_mpm)
    available = [k for k, s in enumerate(field.snapshot_s) if s + latency <= start]
    if not available:
        raise ValueError("No snapshot is available at the start time")
    for k in available:
        fire_map.ingest_snapshot(k)
    truth = TruthTracker(FrontMonitor(field, cell_size_m=config.cell_size_m, base_revisit_s=config.base_revisit_s,
                                      rate_tiers_mpm=config.rate_tiers_mpm), config.radius)
    xmin, ymin, _, _ = field.perimeter(field.snapshot_s[available[-1]]).bounds
    depot = (xmin - config.cell_size_m, ymin - config.cell_size_m)
    env = Environment([Drone(i, *depot) for i in range(config.drones)], speed_mps=config.speed_mps,
                      service_s=config.service_s, sensor_radius_m=config.radius)
    env.time_s = start
    rows, previous, stage = [], {}, 0
    while env.time_s + 1e-9 < stop:
        t = env.time_s
        horizon = min(config.horizon_s, stop - t)
        delivered = []
        for k, s in enumerate(field.snapshot_s):
            if k not in fire_map.snapshots_ingested and s + latency <= t:
                fire_map.ingest_snapshot(k)
                delivered.append(k)
        points = fire_map.points()
        delta = changes(previous, points)
        env.activate(points)
        for k in delivered:
            env.mark_observed(field.snapshot_s[k])
        state = env.state(horizon)
        started = time.perf_counter()
        routes = router(state)
        planning_s = time.perf_counter() - started
        result = env.step(routes, horizon)
        for when, _, i in result.visits:
            fire_map.observe(points[i].x, points[i].y, when)
        positions = {i: (p.x, p.y) for i, p in points.items()}
        satellite = [s for s in field.snapshot_s if t <= s < t + horizon]
        true_points, true_result = truth.score(t, t + horizon, result.visits, positions, satellite)
        believed, actual = set(points), set(true_points)
        rows.append({
            "stage": stage, "time_s": t, "horizon_s": horizon,
            "snapshot_delivered": ",".join(map(str, delivered)),
            "map_points": len(points), "added": len(delta.added), "removed": len(delta.removed),
            "changed": len(delta.changed), "affected_current": len(delta.affected_current),
            "affected_union_fraction": delta.fraction(previous, points) if stage else 1.0,
            "planning_s": planning_s, "visits": len(result.visits),
            "map_objective": result.objective, "map_overdue_count": result.overdue_count,
            "true_front_points": len(true_points), "true_objective": true_result["objective"],
            "true_overdue_count": true_result["overdue_count"], "true_max_gap_s": true_result["max_gap_s"],
            "front_on_map_fraction": len(actual & believed) / max(1, len(actual)),
            "known_burned_fraction": fire_map.known_fraction(t + horizon),
            "total_distance_m": result.total_distance_m, "longest_route_m": result.longest_route_m})
        if record:
            record(stage, fire_map, points, routes)
        previous, stage = points, stage + 1
    return rows
