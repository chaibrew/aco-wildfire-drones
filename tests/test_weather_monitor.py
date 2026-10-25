from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import Point, box

from fire_sim.monitor import FrontMonitor
from fire_sim.spread import ArrivalField, Raster, SpreadModel, interpolate
from fire_sim.terrain import tile_names
from fire_sim.weather import Weather, equilibrium_moisture, ffwi, overpass_utc

T0 = datetime(2018, 11, 8, tzinfo=timezone.utc)
DAY = 43200.0


def payload(hours=48, wind_from=0.0, speed=None):
    stamps = [f"2018-11-08T{h:02d}:00" if h < 24 else f"2018-11-09T{h - 24:02d}:00" for h in range(hours)]
    speed = speed or [5.0] * hours
    return {"hourly": {"time": stamps, "temperature_2m": [25.0] * hours,
                       "relative_humidity_2m": [20.0] * hours, "wind_speed_10m": speed,
                       "wind_direction_10m": [wind_from] * hours}}


def test_ffwi_rises_with_wind_and_dryness():
    assert ffwi(25, 20, 10) > ffwi(25, 20, 2) > ffwi(25, 60, 2)
    assert ffwi(40, 0, 30 / 2.236936) == pytest.approx(100, rel=0.1)
    m = equilibrium_moisture([20, 20, 20, 20], [9.99, 10.0, 49.99, 50.0])
    assert abs(m[0] - m[1]) < 0.5 and abs(m[2] - m[3]) < 0.5


def test_drivers_wind_direction_and_pacing():
    weather = Weather.from_open_meteo(payload(wind_from=0.0), T0)
    d = weather.drivers(3600, 3600 + DAY)
    assert d.wind_to == pytest.approx((0.0, -1.0), abs=1e-9)
    assert d.wind_mps == pytest.approx(5.0)
    assert d.pace_fraction[0] == 0 and d.pace_fraction[-1] == pytest.approx(1)
    assert np.all(np.diff(d.pace_fraction) > 0)
    assert d.pace_times_s[0] == 3600 and d.pace_times_s[-1] == 3600 + DAY


def test_pacing_follows_fire_weather():
    speed = [1.0] * 6 + [12.0] * 6 + [1.0] * 36
    weather = Weather.from_open_meteo(payload(speed=speed), T0)
    d = weather.drivers(0, DAY)
    share = np.interp(12 * 3600, d.pace_times_s, d.pace_fraction) - np.interp(5 * 3600, d.pace_times_s, d.pace_fraction)
    assert share > 0.5


def test_drivers_require_coverage():
    weather = Weather.from_open_meteo(payload(hours=10), T0)
    with pytest.raises(ValueError):
        weather.drivers(0, DAY)


def test_overpass_times():
    assert overpass_utc(datetime(2018, 11, 8, 12), -120) == datetime(2018, 11, 8, 21, 30, tzinfo=timezone.utc)
    assert overpass_utc(datetime(2018, 11, 9, 0), -120) == datetime(2018, 11, 9, 9, 30, tzinfo=timezone.utc)


def test_dem_tile_names():
    assert tile_names(-121.4, 39.8, -121.2, 39.9) == ["Copernicus_DSM_COG_10_N39_00_W122_00_DEM"]
    assert len(tile_names(-122.5, 39.5, -121.5, 40.5)) == 4


@pytest.fixture
def eastward_fire():
    raster = Raster.covering((-3000, -3000, 4000, 3000), 50, 500)
    start = Point(0, 0).buffer(1000, 128)
    geoms = [start, start.union(box(0, -800, 3000, 800))]
    arrival, _ = interpolate(raster, geoms, [0, DAY], model=SpreadModel(noise_sigma=0))
    return ArrivalField(raster, "EPSG:32610", T0, arrival, (0.0, DAY))


def test_active_front_excludes_stalled_perimeter(eastward_fire):
    active = FrontMonitor(eastward_fire, cell_size_m=500).points(DAY / 4)
    everything = FrontMonitor(eastward_fire, cell_size_m=500, active_only=False).points(DAY / 4)
    assert set(active) < set(everything)
    assert all(p.x > -500 for p in active.values())
    assert any(p.x < -500 for p in everything.values())
    assert all(p.risk == 1 for p in everything.values() if p.id not in active)


def test_monitor_points_use_world_grid_ids_and_risk_tiers(eastward_fire):
    monitor = FrontMonitor(eastward_fire, cell_size_m=500, base_revisit_s=900, rate_tiers_mpm=(1.0, 3.0))
    points = monitor.points(DAY / 2)
    for (i, j), p in points.items():
        assert (p.x, p.y) == (i * 500 + 250, j * 500 + 250)
        assert p.revisit_s == 900 / p.risk
    rates = monitor.cell_rates(DAY / 2)
    for i, p in points.items():
        assert p.risk == monitor.risk(rates[i])
    assert list(monitor.risk(np.array([0.5, 1.0, 3.0]) / 60)) == [1, 2, 3]


def test_monitor_rejects_times_outside_observations(eastward_fire):
    monitor = FrontMonitor(eastward_fire)
    with pytest.raises(ValueError):
        monitor.points(DAY + 1)
    assert monitor.points(DAY) == {}
    with pytest.raises(ValueError):
        FrontMonitor(eastward_fire, rate_tiers_mpm=(5.0, 1.0))
