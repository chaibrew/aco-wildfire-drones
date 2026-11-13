from __future__ import annotations

import json
import math
from dataclasses import asdict, fields
from enum import Enum

import numpy as np
from scipy.stats import qmc

from aco_router import ACOParams, BudgetKind, Tau0Mode

SHARED_SPACE: tuple[tuple[str, float, float, bool], ...] = (
    ("alpha", 0.5, 2.0, False),
    ("beta", 0.5, 3.0, False),
    ("urgency_weight", 0.5, 4.0, False),
    ("rho", 0.02, 0.5, True),
    ("tau_min_ratio", 0.001, 0.2, True),
    ("urgency_floor", 0.01, 0.3, True),
)
SHARED_NAMES = tuple(name for name, *_ in SHARED_SPACE)

RHO_HIGH_GRID = (0.2, 0.3, 0.5, 0.7, 0.9)
K_GRID = (2, 5, 10, 20)

# Fixed copy of the pre-tuning values: "defaults" must still mean untuned after config.py changed.
UNTUNED_PARAMS = ACOParams(n_ants=10, max_iterations=15, alpha=1.0, beta=1.0, rho=0.1, urgency_weight=2.0,
                           urgency_floor=0.05, tau_min_ratio=0.01, rho_high=0.5, fast_evaporation_iterations=10)

DECIMALS = 4


def from_unit(u: np.ndarray, space=SHARED_SPACE) -> dict[str, float]:
    out = {}
    for x, (name, lo, hi, log) in zip(u, space):
        v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo))) if log else lo + x * (hi - lo)
        out[name] = round(float(v), DECIMALS)
    return out


def to_unit(values: dict[str, float], space=SHARED_SPACE) -> np.ndarray:
    out = []
    for name, lo, hi, log in space:
        v = values[name]
        x = (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if log else (v - lo) / (hi - lo)
        out.append(min(1.0, max(0.0, x)))
    return np.array(out)


def sobol_candidates(base: ACOParams, n: int = 32, seed: int = 0) -> list[ACOParams]:
    sample = qmc.Sobol(d=len(SHARED_SPACE), scramble=True, seed=seed).random(n)
    return [base] + [base.with_(**from_unit(u)) for u in sample]


def fast_evaporation_grid(base: ACOParams, rho_highs=RHO_HIGH_GRID, ks=K_GRID) -> list[ACOParams]:
    current = (base.rho_high, base.fast_evaporation_iterations)
    out = [base]
    for rh in rho_highs:
        if rh <= base.rho:
            continue
        for k in ks:
            if (rh, k) != current:
                out.append(base.with_(rho_high=rh, fast_evaporation_iterations=k))
    return out


def params_to_dict(params: ACOParams) -> dict:
    d = asdict(params)
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in d.items()}


def params_from_dict(d: dict, base: ACOParams | None = None) -> ACOParams:
    known = {f.name for f in fields(ACOParams)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown ACOParams fields: {sorted(unknown)}")
    vals = dict(d)
    if "budget_kind" in vals:
        vals["budget_kind"] = BudgetKind(vals["budget_kind"])
    if "tau0_mode" in vals:
        vals["tau0_mode"] = Tau0Mode(vals["tau0_mode"])
    return (base or ACOParams()).with_(**vals)


def save_params(path: str, params: ACOParams, **meta) -> None:
    with open(path, "w") as fh:
        json.dump({"params": params_to_dict(params), **meta}, fh, indent=2)
        fh.write("\n")


def load_params(path: str, base: ACOParams | None = None) -> ACOParams:
    with open(path) as fh:
        d = json.load(fh)
    return params_from_dict(d.get("params", d), base)
