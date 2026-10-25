from pathlib import Path

import pytest

from fire_sim.config import load_config
from fire_sim.mission import MissionConfig

EXAMPLE = Path(__file__).parent.parent / "configs" / "camp-3885.toml"


def write(tmp_path, text):
    path = tmp_path / "c.toml"
    path.write_text(text)
    return path


def test_example_config_loads():
    config = load_config(EXAMPLE)
    assert config.data.fire_id == 3885 and config.data.limit == 8
    assert config.mission == MissionConfig(cell_size_m=1000.0, sensor_radius_m=1000.0, drones=6)
    assert config.run.duration_h == 24.0 and config.output.gif


def test_missing_keys_use_defaults(tmp_path):
    config = load_config(write(tmp_path, '[data]\npath = "x.gpkg"\nfire_id = 1\n'))
    assert config.mission == MissionConfig()
    assert config.fire.res_m == 100.0 and config.run.duration_h is None and config.data.year is None
    assert config.mission.radius == config.mission.cell_size_m


@pytest.mark.parametrize("text", [
    '[data]\npath = "x"\nfire_id = 1\n[mission]\ncell_size = 500\n',
    '[data]\npath = "x"\nfire_id = 1\n[missions]\n',
    '[mission]\ndrones = 3\n',
    '[data]\npath = "x"\n',
])
def test_bad_configs_are_rejected(tmp_path, text):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, text))
