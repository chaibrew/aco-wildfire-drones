from dataclasses import dataclass
from datetime import datetime

from shapely.geometry.base import BaseGeometry

CellId = tuple[int, int]


@dataclass(frozen=True)
class Perimeter:
    timestamp: datetime
    geometry: BaseGeometry


@dataclass(frozen=True)
class MonitoringPoint:
    id: CellId
    x: float
    y: float
    risk: int
    revisit_s: float


@dataclass(frozen=True)
class ChangeSet:
    added: frozenset[CellId]
    removed: frozenset[CellId]
    changed: frozenset[CellId]

    @property
    def affected_current(self) -> frozenset[CellId]:
        return self.added | self.changed

    def fraction(self, previous: dict, current: dict) -> float:
        return len(self.added | self.removed | self.changed) / max(1, len(previous.keys() | current.keys()))


def changes(previous, current):
    old, new = previous.keys(), current.keys()
    altered = {i for i in old & new if previous[i] != current[i]}
    return ChangeSet(frozenset(new - old), frozenset(old - new), frozenset(altered))
