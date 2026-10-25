from fire_sim import Drone, Environment, Grid, evaluate
from fire_sim.grid import perimeter_changes
from fire_sim.scenarios import synthetic_sequence
from fire_sim.simulation import smoke_router


def main():
    grid = Grid()
    env = Environment([Drone(i, -2000, -2000) for i in range(3)])
    previous_points, previous_geometry = {}, None
    for frame in synthetic_sequence(3):
        points = grid.points(frame.geometry)
        delta = perimeter_changes(grid, previous_geometry or frame.geometry,
                                  frame.geometry, previous_points, points)
        env.activate(points)
        state = env.state(horizon_s=600)
        routes = smoke_router(state)
        predicted = evaluate(state, routes)
        actual = env.step(routes, horizon_s=600)
        assert predicted.objective == actual.objective
        print(f"t={env.time_s:.0f}s active={len(points)} affected={len(delta.affected_current)} J={actual.objective:.3f}")
        previous_points, previous_geometry = points, frame.geometry


if __name__ == "__main__":
    main()
