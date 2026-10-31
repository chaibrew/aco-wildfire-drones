import numpy as np

from aco_router.config import ACOParams
from aco_router.model import Drone, MissionInstance, Point

REVISIT_S = (900.0, 450.0, 300.0)  # fire_sim: 900 / risk


def random_instance(seed: int, n_points: int = 6, n_drones: int = 2, time: float = 0.0, horizon: float = 600.0,
                    spread_m: float = 1500.0):
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n_points):
        x, y = rng.uniform(-spread_m, spread_m, 2)
        q = REVISIT_S[int(rng.integers(3))]
        pts.append(Point(i, float(x), float(y), q, float(time - rng.uniform(0, 600))))
    drones = tuple(Drone(f"d{k}", 0.0, 0.0, 15.0) for k in range(n_drones))
    return MissionInstance(tuple(pts), drones, time=time, horizon=horizon, service_s=5.0)


def tiny_instance(seed: int):
    rng = np.random.default_rng(seed)
    pts = tuple(
        Point(i, *map(float, rng.uniform(-1200, 1200, 2)), float(rng.choice([60.0, 120.0, 240.0])),
              last_seen=float(-rng.uniform(0, 200)))
        for i in range(4)
    )
    return MissionInstance(pts, (Drone("a", 0, 0, 15), Drone("b", 0, 0, 15)), time=0.0, horizon=220.0, service_s=5.0)


SMALL_PARAMS = ACOParams(n_ants=10, max_iterations=30)
