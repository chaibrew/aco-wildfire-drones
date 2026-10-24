import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

from fire_sim.feds import inspect_file, load_feds


def write_source(tmp_path, geometries=None, years=None):
    path = tmp_path / "test.gpkg"
    frame = gpd.GeoDataFrame({"time": ["2018-07-02T00:00:00", "2018-07-01T12:00:00"],
                              "fireID": [7, 7], "year": years or [2018, 2018]},
                             geometry=geometries or [box(-121, 39, -120.98, 39.02), box(-121, 39, -120.99, 39.01)],
                             crs="EPSG:4326")
    frame.to_file(path, layer="perimeter", driver="GPKG")
    return path


def test_real_gpkg_roundtrip_sort_projection_and_metadata(tmp_path):
    path = write_source(tmp_path)
    sequence, metadata = load_feds(path, fire_id=7)
    assert sequence[0].timestamp < sequence[1].timestamp
    assert 500 < sequence[0].geometry.bounds[2] - sequence[0].geometry.bounds[0] < 2000
    assert metadata["target_crs"] == "EPSG:32610"
    assert len(metadata["source_sha256"]) == 64
    assert inspect_file(path)["fires"][0]["snapshots"] == 2


def test_multi_year_id_requires_year(tmp_path):
    path = write_source(tmp_path, years=[2018, 2019])
    with pytest.raises(ValueError, match="repeats"):
        load_feds(path, fire_id=7)
    assert len(load_feds(path, fire_id=7, year=2018)[0]) == 1


def test_invalid_geometry_explicit_repair(tmp_path):
    crossed = Polygon([(-121, 39), (-120.98, 39.02), (-121, 39.02), (-120.98, 39), (-121, 39)])
    path = write_source(tmp_path, [crossed, box(-121, 39, -120.99, 39.01)])
    with pytest.raises(ValueError, match="Invalid perimeter"):
        load_feds(path, fire_id=7)
    sequence, metadata = load_feds(path, fire_id=7, repair=True)
    assert metadata["geometry_repairs"] == 1
    assert all(p.geometry.is_valid for p in sequence)


def test_missing_fire_and_nonmetric_projection_rejected(tmp_path):
    path = write_source(tmp_path)
    with pytest.raises(ValueError, match="No records"):
        load_feds(path, fire_id=99)
    for crs in ("EPSG:4326", "EPSG:2227"):
        with pytest.raises(ValueError, match="metre"):
            load_feds(path, fire_id=7, target_crs=crs)
