from datetime import datetime, timezone

import numpy as np
import pytest
from shapely.geometry import Point, box

from fire_sim.spread import ArrivalField, IntervalDrivers, Raster, SpreadModel, interpolate

CALM = SpreadModel(noise_sigma=0.0)
DAY = 43200.0


@pytest.fixture
def raster():
    return Raster.covering((-3000, -3000, 3000, 3000), 50, 500)


def discs(*radii, centre=(0, 0)):
    return [Point(*centre).buffer(r, 128) for r in radii]


def at(raster, values, x, y):
    return values[int((y - raster.y0) // raster.res), int((x - raster.x0) // raster.res)]


def ring(raster, radius, width=25):
    x, y = raster.centres()
    return np.abs(np.hypot(x, y) - radius) < width


def test_raster_origin_nests_in_monitoring_grid(raster):
    assert raster.x0 % 500 == 0 and raster.y0 % 500 == 0
    assert raster.mask(box(-100, -100, 100, 100)).sum() == 16


def test_calm_concentric_growth_is_linear_in_distance(raster):
    arrival, info = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM)
    for radius in (1100, 1500, 1900):
        expected = (radius - 1000) / 1000 * DAY
        assert np.mean(arrival[ring(raster, radius)]) == pytest.approx(expected, abs=0.02 * DAY)
    assert info[0]["spot_fires"] == 0


@pytest.mark.parametrize("model,drivers,elevation", [
    (CALM, None, None),
    (SpreadModel(noise_sigma=0.5), [IntervalDrivers((1.0, 0.0), 10.0)] * 2, None),
    (SpreadModel(), None, "slope"),
])
def test_every_snapshot_is_reproduced_exactly(raster, model, drivers, elevation):
    x, _ = raster.centres()
    elev = 0.3 * x if elevation else None
    geoms = discs(800, 1500) + [Point(300, 0).buffer(2400, 128)]
    times = [0, DAY, 2 * DAY]
    arrival, _ = interpolate(raster, geoms, times, model=model, drivers=drivers, elevation=elev, seed=3)
    cumulative = np.zeros(raster.shape, bool)
    for geom, t in zip(geoms, times):
        cumulative |= raster.mask(geom)
        assert np.array_equal(arrival <= t, cumulative)
    band = np.isfinite(arrival) & (arrival > 0)
    assert (arrival[band] > 0).all() and (arrival[band] <= 2 * DAY).all()


def test_wind_burns_downwind_side_first(raster):
    x, _ = raster.centres()
    middle = ring(raster, 1500)
    arrival, _ = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM,
                             drivers=[IntervalDrivers((1.0, 0.0), 8.0)])
    assert arrival[middle & (x > 1400)].mean() < arrival[middle & (x < -1400)].mean() - 0.1 * DAY


def test_steep_ground_is_crossed_quickly(raster):
    # Snapshots fix where the fire got to, so slope reshapes timing along the path:
    # a steep inner half of the band (uphill outwards) burns early, the flat outer half late.
    x, y = raster.centres()
    elevation = 0.5 * np.clip(np.hypot(x, y) - 1000, 0, 500)
    arrival, _ = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM, elevation=elevation)
    assert arrival[ring(raster, 1500)].mean() < 0.3 * DAY
    assert np.array_equal(arrival <= DAY, raster.mask(discs(2000)[0]))


def test_pacing_curve_moves_growth_in_time(raster):
    clock = tuple(np.linspace(0, DAY, 25))
    late = tuple((np.linspace(0, 1, 25)) ** 3)  # most growth near the end
    uniform, _ = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM)
    paced, _ = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM,
                           drivers=[IntervalDrivers(pace_times_s=clock, pace_fraction=late)])
    middle = ring(raster, 1500)
    assert paced[middle].mean() > uniform[middle].mean() + 0.1 * DAY
    assert np.array_equal(paced <= DAY, uniform <= DAY)


def test_seed_controls_heterogeneity(raster):
    geoms = discs(1000, 2000)
    noisy = SpreadModel(noise_sigma=0.5)
    a, _ = interpolate(raster, geoms, [0, DAY], model=noisy, seed=1)
    b, _ = interpolate(raster, geoms, [0, DAY], model=noisy, seed=1)
    c, _ = interpolate(raster, geoms, [0, DAY], model=noisy, seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_detached_spot_fire_ignites_within_interval(raster):
    later = discs(1000)[0].union(Point(2300, 2300).buffer(300))
    arrival, info = interpolate(raster, [discs(800)[0], later], [0, DAY], model=CALM)
    assert info[0]["spot_fires"] == 1
    spot = at(raster, arrival, 2300, 2300)
    assert 0 < spot < DAY


def test_unburned_island_fills_in(raster):
    annulus = discs(1500)[0].difference(Point(0, 0).buffer(500))
    arrival, info = interpolate(raster, [annulus, discs(1500)[0]], [0, DAY], model=CALM)
    assert info[0]["enclosed_islands"] == 1
    assert 0 < at(raster, arrival, 0, 0) <= DAY
    assert at(raster, arrival, 450, 0) < at(raster, arrival, 25, 25)


def test_invalid_times_rejected(raster):
    with pytest.raises(ValueError):
        interpolate(raster, discs(1000, 2000), [DAY, 0])
    with pytest.raises(ValueError):
        interpolate(raster, discs(1000, 2000), [0])


def test_field_rate_perimeter_and_round_trip(raster, tmp_path):
    arrival, _ = interpolate(raster, discs(1000, 2000), [0, DAY], model=CALM)
    field = ArrivalField(raster, "EPSG:32610", datetime(2020, 1, 1, tzinfo=timezone.utc), arrival, (0.0, DAY))
    assert np.nanmedian(field.rate_mps) == pytest.approx(1000 / DAY, rel=0.05)
    assert field.perimeter(DAY / 2).area == pytest.approx(np.pi * 1500 ** 2, rel=0.03)
    field.save(tmp_path / "f")
    again = ArrivalField.load(tmp_path / "f")
    assert np.array_equal(again.arrival_s, field.arrival_s)
    assert again.raster == raster and again.t0 == field.t0 and again.snapshot_s == field.snapshot_s
    assert field.next_snapshot(10) == DAY and field.next_snapshot(DAY) == DAY
