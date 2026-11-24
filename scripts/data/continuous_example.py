from fire_sim.fire import load_fire
from fire_sim.mission import MissionConfig, run_mission
from fire_sim.simulation import evaluate, smoke_router


class Router:
    def __init__(self):
        self.previous = set()

    def __call__(self, state):
        current = set(state.points)
        added, removed = current - self.previous, self.previous - current
        routes = smoke_router(state)
        print(f"t={state.time_s / 3600:4.1f}h points={len(current):3d} +{len(added):3d} -{len(removed):3d} "
              f"J_map={evaluate(state, routes).objective:.2f}")
        self.previous = current
        return routes


def main():
    field, _ = load_fire("data/raw/LargeFires_2018.gpkg", fire_id=3885, year=2018, limit=8)
    rows, _ = run_mission(field, Router(), MissionConfig(cell_size_m=1000, drones=6), stop_s=2 * 3600)
    print(f"true-front J at end: {rows[-1]['true_objective']:.2f}, front on map: {rows[-1]['front_on_map_fraction']:.0%}")


if __name__ == "__main__":
    main()
