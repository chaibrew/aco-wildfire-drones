from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class StrategyName(str, Enum):
    RETENTION = "retention"
    FAST_EVAPORATION = "fast_evaporation"
    SELECTIVE_RESET = "selective_reset"
    FULL_RESET = "full_reset"


class ChangeScope(str, Enum):
    NONE = "none"
    LOCAL = "local"
    WIDESPREAD = "widespread"


class BudgetKind(str, Enum):
    ITERATIONS = "iterations"
    SECONDS = "seconds"


class Tau0Mode(str, Enum):
    CONSTANT = "constant"
    NEAREST_NEIGHBOUR = "nearest_neighbour"


# defaults match fire_sim's MissionConfig
PLANNING_HORIZON_S: float = 600.0
DRONE_SPEED_MS: float = 15.0
SERVICE_TIME_S: float = 5.0

COUNT_Q_CHANGE_AS_MODIFIED: bool = True
POSITION_TOLERANCE_M: float = 1.0
Q_RELATIVE_TOLERANCE: float = 1e-6
LOCAL_CHANGE_MAX_FRACTION: float = 0.5

DISTANCE_EPS_M: float = 1e-6
COST_EPS: float = 1e-9
HORIZON_TOLERANCE_S: float = 1e-9


@dataclass(frozen=True)
class ACOParams:
    # Tuned defaults (docs/tuning.md); pre-tuning values are in tuning.space.UNTUNED_PARAMS.
    n_ants: int = 20
    alpha: float = 0.73
    beta: float = 2.49
    rho: float = 0.36
    deposit_q: float = 1.0
    tau0_mode: Tau0Mode = Tau0Mode.NEAREST_NEIGHBOUR
    tau0_constant: float = 1.0

    rho_high: float = 0.5
    fast_evaporation_iterations: int = 10

    urgency_weight: float = 1.93
    urgency_floor: float = 0.12

    use_bounds: bool = True
    tau_min_ratio: float = 0.024

    budget_kind: BudgetKind = BudgetKind.ITERATIONS
    max_iterations: int = 60
    time_budget_s: float = 10.0

    def __post_init__(self) -> None:
        if self.n_ants < 1:
            raise ValueError("n_ants must be >= 1")
        for name in ("rho", "rho_high"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.urgency_floor <= 0:
            raise ValueError("urgency_floor must be > 0")

    def with_(self, **changes) -> "ACOParams":
        return replace(self, **changes)


DEFAULT_ACO_PARAMS = ACOParams()
