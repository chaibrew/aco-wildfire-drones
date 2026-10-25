import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .mission import MissionConfig


@dataclass(frozen=True)
class DataConfig:
    path: str
    fire_id: int | str
    year: int | None = None
    limit: int | None = None
    crs: str | None = None


@dataclass(frozen=True)
class FireConfig:
    res_m: float = 100.0
    seed: int = 0
    noise_sigma: float = 0.3
    weather: bool = True
    terrain: bool = True
    time_convention: str = "overpass"


@dataclass(frozen=True)
class RunConfig:
    start_h: float = 0.0
    duration_h: float | None = None


@dataclass(frozen=True)
class OutputConfig:
    fire_dir: str = "outputs/fire"
    mission_dir: str = "outputs/mission"
    gif: bool = False
    gif_step_h: float = 0.5
    gif_every: int = 3
    validate: bool = False


@dataclass(frozen=True)
class Config:
    data: DataConfig
    fire: FireConfig = field(default_factory=FireConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    run: RunConfig = field(default_factory=RunConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    source: str = ""


SECTIONS = {"data": DataConfig, "fire": FireConfig, "mission": MissionConfig,
            "run": RunConfig, "output": OutputConfig}


def _section(name, kind, values):
    allowed = {f.name for f in fields(kind)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"[{name}] unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    if kind is MissionConfig and "rate_tiers_mpm" in values:
        values = {**values, "rate_tiers_mpm": tuple(values["rate_tiers_mpm"])}
    try:
        return kind(**values)
    except TypeError as error:
        raise ValueError(f"[{name}] {error}") from None


def load_config(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}; allowed: {sorted(SECTIONS)}")
    if "data" not in raw:
        raise ValueError("[data] section is required")
    parts = {name: _section(name, kind, raw.get(name, {})) for name, kind in SECTIONS.items()}
    return Config(**parts, source=str(path))
