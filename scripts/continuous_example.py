"""Planner hook on a continuous fire: replace smoke_router with ACO/greedy/VNS.

Run after `fire-sim fire data/raw/LargeFires_2018.gpkg --fire-id 3885 --year 2018 --limit 8`
(or let load_fire build and cache the field on first use).
"""

from fire_sim import Drone, Environment, changes, evaluate
from fire_sim.fire import load_fire
from fire_sim.monitor import FrontMonitor
from fire_sim.simulation import smoke_router

HORIZON_S = 600


def main():
    field, _ = load_fire("data/raw/LargeFires_2018.gpkg", fire_id=3885, year=2018, limit=8, pacing_exponent=2.0)
    monitor = FrontMonitor(field, cell_size_m=1000)
    xmin, ymin, _, _ = field.perimeter(field.start_s).bounds
    env = Environment([Drone(i, xmin, ymin) for i in range(6)])
    env.time_s = field.start_s
    previous = {}
    for _ in range(12):  # two hours
        points = monitor.points(env.time_s)
        delta = changes(previous, points)
        env.activate(points)
        state = env.state(HORIZON_S)
        # ACO hook: remap pheromone by stable cell IDs; delta.affected_current is C_t
        # for selective reset, delta.removed must have its edges deleted.
        routes = smoke_router(state)
        assert evaluate(state, routes).objective == env.step(routes, HORIZON_S).objective
        print(f"t={state.time_s / 3600:4.1f}h points={len(points):3d} "
              f"C_t={len(delta.affected_current):3d} removed={len(delta.removed):2d}")
        previous = points


if __name__ == "__main__":
    main()
