from __future__ import annotations

import math
from dataclasses import replace

from .config import (
    COUNT_Q_CHANGE_AS_MODIFIED,
    LOCAL_CHANGE_MAX_FRACTION,
    POSITION_TOLERANCE_M,
    Q_RELATIVE_TOLERANCE,
    ChangeScope,
)
from .model import ChangeEvent, MissionInstance


def diff_instances(
    old: MissionInstance,
    new: MissionInstance,
    *,
    count_q_change: bool = COUNT_Q_CHANGE_AS_MODIFIED,
    position_tolerance_m: float = POSITION_TOLERANCE_M,
    q_relative_tolerance: float = Q_RELATIVE_TOLERANCE,
    local_max_fraction: float = LOCAL_CHANGE_MAX_FRACTION,
) -> ChangeEvent:
    old_ids = set(old.point_ids)
    new_ids = set(new.point_ids)
    added = new_ids - old_ids
    removed = old_ids - new_ids
    modified = set()
    for pid in old_ids & new_ids:
        a, b = old.point(pid), new.point(pid)
        moved = math.hypot(b.x - a.x, b.y - a.y) > position_tolerance_m
        q_changed = abs(b.q - a.q) > q_relative_tolerance * max(a.q, b.q)
        if moved or (count_q_change and q_changed):
            modified.add(pid)
    union = old_ids | new_ids
    changed = len(added) + len(removed) + len(modified)
    fraction = changed / len(union) if union else 0.0
    if changed == 0:
        scope = ChangeScope.NONE
    elif fraction <= local_max_fraction:
        scope = ChangeScope.LOCAL
    else:
        scope = ChangeScope.WIDESPREAD
    return ChangeEvent(frozenset(added), frozenset(removed), frozenset(modified), fraction, scope)


def with_neighbourhood(change: ChangeEvent, old: MissionInstance, new: MissionInstance,
                       radius_m: float) -> ChangeEvent:
    if radius_m <= 0 or change.is_empty:
        return change
    centres = [(new.point(i).x, new.point(i).y) for i in change.added | change.modified]
    centres += [(old.point(i).x, old.point(i).y) for i in change.removed]
    near = set()
    for p in new.points:
        if any(math.hypot(p.x - x, p.y - y) <= radius_m + 1e-6 for x, y in centres):
            near.add(p.id)
    return replace(change, neighbourhood=frozenset(near))
