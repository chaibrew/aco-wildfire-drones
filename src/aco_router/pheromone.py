from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Sequence

import numpy as np

from .model import MissionInstance


@dataclass(frozen=True)
class DroneNode:
    drone_id: Hashable


def node_keys(instance: MissionInstance) -> tuple[Hashable, ...]:
    return tuple(DroneNode(d.id) for d in instance.drones) + instance.point_ids


class PheromoneMatrix:
    def __init__(self, keys: Sequence[Hashable], tau0: float, values: np.ndarray | None = None):
        self.keys = tuple(keys)
        self.index = {k: i for i, k in enumerate(self.keys)}
        if len(self.index) != len(self.keys):
            raise ValueError("duplicate pheromone keys")
        n = len(self.keys)
        if values is None:
            self.tau = np.full((n, n), float(tau0))
        else:
            if values.shape != (n, n):
                raise ValueError(f"values shape {values.shape} does not match {n} keys")
            self.tau = np.array(values, dtype=float, copy=True)

    def __len__(self) -> int:
        return len(self.keys)

    def copy(self) -> "PheromoneMatrix":
        return PheromoneMatrix(self.keys, 0.0, self.tau)

    def get(self, a: Hashable, b: Hashable) -> float:
        return float(self.tau[self.index[a], self.index[b]])

    def remapped(self, new_keys: Sequence[Hashable], tau0: float) -> "PheromoneMatrix":
        new = PheromoneMatrix(new_keys, tau0)
        shared_new, shared_old = [], []
        for i, key in enumerate(new.keys):
            j = self.index.get(key)
            if j is not None:
                shared_new.append(i)
                shared_old.append(j)
        if shared_new:
            new.tau[np.ix_(shared_new, shared_new)] = self.tau[np.ix_(shared_old, shared_old)]
        return new

    def reset_all(self, tau0: float) -> None:
        self.tau.fill(tau0)

    def reset_touching(self, keys: Iterable[Hashable], tau0: float) -> None:
        idx = [self.index[k] for k in keys if k in self.index]
        if idx:
            self.tau[idx, :] = tau0
            self.tau[:, idx] = tau0

    def evaporate(self, rho: float) -> None:
        self.tau *= 1.0 - rho

    def deposit(self, rows: np.ndarray, cols: np.ndarray, amount: float) -> None:
        np.add.at(self.tau, (rows, cols), amount)

    def clip(self, low: float, high: float) -> None:
        np.clip(self.tau, low, high, out=self.tau)
