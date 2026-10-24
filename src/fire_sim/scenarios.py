"""Synthetic smoke cases and controlled geometric growth experiments."""

import math
import random
from datetime import datetime, timedelta, timezone

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .models import Perimeter


def synthetic_sequence(steps=6):
    if steps < 1:
        raise ValueError("steps must be positive")
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return [Perimeter(start + timedelta(hours=12 * k), Polygon([
        ((1500 + 90 * k) * (1 + 0.10 * math.sin(5 * a)) * math.cos(a),
         (1100 + 60 * k) * (1 + 0.12 * math.sin(3 * a)) * math.sin(a))
        for a in [2 * math.pi * j / 160 for j in range(160)]
    ])) for k in range(steps)]


def perturb(perimeter, *, locality="local", radius_m=400.0, patches=4, seed=42):
    """Add circular growth patches along exteriors. Actual changed fraction is measured.

    'local' uses a contiguous tenth of boundary arc length on the largest polygon;
    'distributed' spaces patches over the total exterior arc length of all parts.
    Radius/patch count control geometry, not an exact monitoring-point fraction.
    """
    if locality not in {"local", "distributed"}:
        raise ValueError("locality must be local or distributed")
    if not math.isfinite(radius_m) or radius_m <= 0 or patches < 1:
        raise ValueError("Positive radius and patch count required")
    parts = list(perimeter.geoms) if perimeter.geom_type == "MultiPolygon" else [perimeter]
    if locality == "local":
        parts = [max(parts, key=lambda p: p.area)]
    rings = [p.exterior for p in parts]
    length = sum(r.length for r in rings)
    phase = random.Random(seed).random()
    additions = []
    for j in range(patches):
        fraction = (phase + (0.1 if locality == "local" else 1.0) * j / patches) % 1
        offset = fraction * length
        for ring in rings:
            if offset <= ring.length:
                center = ring.interpolate(offset)
                additions.append(Point(center.x, center.y).buffer(radius_m))
                break
            offset -= ring.length
    return unary_union([perimeter, *additions])
