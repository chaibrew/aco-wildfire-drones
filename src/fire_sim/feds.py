"""Adapter for the published 2012–2020 FEDS Largefire GeoPackages."""

import hashlib
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import CRS
from shapely import make_valid
from shapely.ops import unary_union

from .models import Perimeter


def inspect_file(path):
    layers = pyogrio.list_layers(path).tolist()
    info = {"layers": layers}
    if "perimeter" in [layer[0] for layer in layers]:
        frame = pyogrio.read_dataframe(path, layer="perimeter", read_geometry=False)
        info["columns"] = list(frame.columns)
        if {"year", "fireID", "time"}.issubset(frame.columns):
            info["fires"] = (frame.groupby(["year", "fireID"])
                             .agg(snapshots=("time", "nunique"), start=("time", "min"), end=("time", "max"))
                             .reset_index().to_dict(orient="records"))
    return info


def _polygonal(geom):
    if geom is None or geom.is_empty:
        raise ValueError("Null or empty perimeter geometry")
    if geom.geom_type in {"Polygon", "MultiPolygon"}:
        return geom
    if geom.geom_type == "GeometryCollection":
        parts = [g for g in geom.geoms if g.geom_type in {"Polygon", "MultiPolygon"} and not g.is_empty]
        if parts:
            return unary_union(parts)
    raise ValueError(f"Expected polygonal perimeter, got {geom.geom_type}")


def load_feds(path, *, fire_id, year=None, target_crs=None, repair=False, limit=None):
    """Return a sorted sequence in one metric CRS plus provenance.

    The published time field is a half-day label without a timezone. Preserve it
    as supplied: this adapter does not claim labels are exact UTC detections.
    """
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    frame = gpd.read_file(path, layer="perimeter", engine="pyogrio")
    if frame.crs is None:
        raise ValueError("FEDS file has no CRS; supply a correctly tagged source file")
    required = {"time", "fireID", "year"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing FEDS columns: {required - set(frame.columns)}")
    frame = frame[frame.fireID.astype(str) == str(fire_id)].copy()
    if year is not None:
        frame = frame[frame.year == year].copy()
    if frame.empty:
        raise ValueError(f"No records for fireID={fire_id}, year={year}")
    if frame.year.nunique() != 1:
        raise ValueError("fireID repeats across years; select --year")
    source_crs = str(frame.crs)
    frame["parsed_time"] = pd.to_datetime(frame.time, errors="raise", format="ISO8601")
    if frame.parsed_time.isna().any():
        raise ValueError("Missing timestamps")
    frame = frame.sort_values("parsed_time")
    if limit is not None:
        frame = frame[frame.parsed_time.isin(frame.parsed_time.drop_duplicates().iloc[:limit])].copy()
    repairs = 0
    geometries = []
    for geom in frame.geometry:
        if geom is None or geom.is_empty:
            raise ValueError("Null or empty perimeter geometry")
        if not geom.is_valid:
            if not repair:
                raise ValueError("Invalid perimeter; use --repair to enable recorded make_valid repairs")
            geom = make_valid(geom)
            repairs += 1
        geometries.append(_polygonal(geom))
    frame.geometry = geometries
    # First snapshot only: avoid using future fire extent to choose the projection.
    crs = CRS.from_user_input(target_crs or frame.iloc[:1].estimate_utm_crs())
    if not crs.is_projected or not crs.axis_info or any(abs(a.unit_conversion_factor - 1) > 1e-12 for a in crs.axis_info):
        raise ValueError("Target CRS must be projected with metre units")
    frame = frame.to_crs(crs)
    sequence = [Perimeter(stamp.to_pydatetime(), _polygonal(unary_union(group.geometry.tolist())))
                for stamp, group in frame.groupby("parsed_time", sort=True)]
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {"source": str(path.resolve()), "source_sha256": sha,
                  "source_crs": source_crs, "target_crs": crs.to_string(),
                  "year": int(frame.year.iloc[0]), "fire_id": str(fire_id),
                  "layer": "perimeter", "geometry_repairs": repairs,
                  "timestamp_convention": "source half-day labels; timezone unspecified",
                  "duplicate_rows_unioned": len(frame) - len(sequence),
                  "dataset_doi": "https://doi.org/10.6084/m9.figshare.c.5601537.v1"}
    return sequence, provenance
