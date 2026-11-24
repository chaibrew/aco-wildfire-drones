from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Hashable, Iterable, Mapping

from .config import (
    DRONE_SPEED_MS,
    PLANNING_HORIZON_S,
    SERVICE_TIME_S,
    ChangeScope,
)

PointId = Hashable
DroneId = Hashable


@dataclass(frozen=True)
class Point:
    id: PointId
    x: float
    y: float
    q: float
    last_seen: float | None = None

    def __post_init__(self) -> None:
        if not self.q > 0:
            raise ValueError(f"point {self.id!r}: q must be > 0, got {self.q}")


@dataclass(frozen=True)
class Drone:
    id: DroneId
    x: float
    y: float
    speed: float = DRONE_SPEED_MS

    def __post_init__(self) -> None:
        if not self.speed > 0:
            raise ValueError(f"drone {self.id!r}: speed must be > 0")


@dataclass(frozen=True, eq=False)
class MissionInstance:
    points: tuple[Point, ...]
    drones: tuple[Drone, ...]
    time: float = 0.0
    horizon: float = PLANNING_HORIZON_S
    stage: int = 0
    service_s: float = SERVICE_TIME_S
    coverage: Mapping[PointId, tuple[PointId, ...]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "drones", tuple(self.drones))
        if not self.drones:
            raise ValueError("an instance needs at least one drone")
        if not self.horizon > 0:
            raise ValueError("horizon must be > 0")
        if self.service_s < 0:
            raise ValueError("service_s must be >= 0")
        _require_unique((p.id for p in self.points), "point")
        _require_unique((d.id for d in self.drones), "drone")
        for p in self.points:
            if p.last_seen is not None and p.last_seen > self.time + 1e-9:
                raise ValueError(f"point {p.id!r}: last_seen {p.last_seen} is after stage time {self.time}")

    @cached_property
    def points_by_id(self) -> Mapping[PointId, Point]:
        return {p.id: p for p in self.points}

    @cached_property
    def drones_by_id(self) -> Mapping[DroneId, Drone]:
        return {d.id: d for d in self.drones}

    @property
    def point_ids(self) -> tuple[PointId, ...]:
        return tuple(p.id for p in self.points)

    @property
    def drone_ids(self) -> tuple[DroneId, ...]:
        return tuple(d.id for d in self.drones)

    @property
    def end_time(self) -> float:
        return self.time + self.horizon

    def point(self, point_id: PointId) -> Point:
        return self.points_by_id[point_id]

    def drone(self, drone_id: DroneId) -> Drone:
        return self.drones_by_id[drone_id]

    def last_seen(self, point: Point) -> float:
        return self.time if point.last_seen is None else point.last_seen

    def observed_by(self, point_id: PointId) -> tuple[PointId, ...]:
        if self.coverage is None:
            return (point_id,)
        return tuple(self.coverage.get(point_id, (point_id,)))


@dataclass(frozen=True, eq=False)
class Plan:
    routes: Mapping[DroneId, tuple[PointId, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "routes", {d: tuple(r) for d, r in dict(self.routes).items()})

    @classmethod
    def empty(cls, instance: MissionInstance) -> "Plan":
        return cls({d: () for d in instance.drone_ids})

    def route(self, drone_id: DroneId) -> tuple[PointId, ...]:
        return self.routes.get(drone_id, ())

    def to_dict(self) -> dict[DroneId, list[PointId]]:
        return {d: list(r) for d, r in self.routes.items()}

    def for_drones(self, drone_ids: Iterable[DroneId]) -> dict[DroneId, list[PointId]]:
        return {d: list(self.route(d)) for d in drone_ids}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Plan):
            return NotImplemented
        mine = {d: r for d, r in self.routes.items() if r}
        theirs = {d: r for d, r in other.routes.items() if r}
        return mine == theirs

    def __repr__(self) -> str:
        return f"Plan({self.to_dict()!r})"


@dataclass(frozen=True)
class ChangeEvent:
    added: frozenset = frozenset()
    removed: frozenset = frozenset()
    modified: frozenset = frozenset()
    fraction: float = 0.0
    scope: ChangeScope = ChangeScope.NONE
    neighbourhood: frozenset = frozenset()

    @property
    def affected(self) -> frozenset:
        return frozenset(self.added | self.modified)

    @property
    def reset_scope(self) -> frozenset:
        return self.affected | self.neighbourhood

    @property
    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified)

    @classmethod
    def none(cls) -> "ChangeEvent":
        return cls()


@dataclass(frozen=True)
class Metrics:
    objective: float
    margin: float
    total_overdue: float
    overdue_count: int
    mean_gap_s: float
    max_gap_s: float
    total_distance_m: float
    longest_route_m: float
    gaps_s: Mapping[PointId, float] = field(default_factory=dict, compare=False, repr=False)

    @property
    def score(self) -> tuple[float, float, float]:
        return (self.margin, self.total_overdue, self.total_distance_m)

    @property
    def cost(self) -> float:
        n = max(1, len(self.gaps_s))
        return self.margin + self.total_overdue / n


@dataclass(frozen=True)
class TraceEntry:
    iteration: int
    elapsed_s: float
    best_objective: float
    best_margin: float
    best_total_overdue: float = 0.0


@dataclass(frozen=True, eq=False)
class PlanResult:
    plan: Plan
    metrics: Metrics
    router: str
    runtime_s: float
    iterations: int = 0
    trace: tuple[TraceEntry, ...] = ()

    @property
    def objective(self) -> float:
        return self.metrics.objective


def _require_unique(ids: Iterable[Hashable], kind: str) -> None:
    seen: set = set()
    for i in ids:
        if i in seen:
            raise ValueError(f"duplicate {kind} id {i!r}")
        seen.add(i)
