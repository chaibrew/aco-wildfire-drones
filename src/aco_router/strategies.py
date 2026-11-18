from __future__ import annotations

from abc import ABC

from .config import ACOParams, StrategyName
from .model import ChangeEvent
from .pheromone import PheromoneMatrix


class PheromoneStrategy(ABC):
    name: StrategyName

    def on_change(self, pheromone: PheromoneMatrix, change: ChangeEvent, tau0: float) -> None:
        pass

    def evaporation_rate(self, params: ACOParams, iterations_since_change: int | None) -> float:
        return params.rho

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Retention(PheromoneStrategy):
    name = StrategyName.RETENTION


class FullReset(PheromoneStrategy):
    name = StrategyName.FULL_RESET

    def on_change(self, pheromone: PheromoneMatrix, change: ChangeEvent, tau0: float) -> None:
        pheromone.reset_all(tau0)


class SelectiveReset(PheromoneStrategy):
    name = StrategyName.SELECTIVE_RESET

    def on_change(self, pheromone: PheromoneMatrix, change: ChangeEvent, tau0: float) -> None:
        pheromone.reset_touching(change.reset_scope, tau0)


class FastEvaporation(PheromoneStrategy):
    name = StrategyName.FAST_EVAPORATION

    def evaporation_rate(self, params: ACOParams, iterations_since_change: int | None) -> float:
        if iterations_since_change is not None and iterations_since_change < params.fast_evaporation_iterations:
            return params.rho_high
        return params.rho


STRATEGIES: dict[StrategyName, type[PheromoneStrategy]] = {
    StrategyName.RETENTION: Retention,
    StrategyName.FAST_EVAPORATION: FastEvaporation,
    StrategyName.SELECTIVE_RESET: SelectiveReset,
    StrategyName.FULL_RESET: FullReset,
}


def make_strategy(name: StrategyName | str | PheromoneStrategy) -> PheromoneStrategy:
    if isinstance(name, PheromoneStrategy):
        return name
    return STRATEGIES[StrategyName(name)]()
