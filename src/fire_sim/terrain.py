import hashlib
import json
import math
from pathlib import Path

import numpy as np
from pyproj import Transformer

from .spread import Raster

TILE_URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"


def tile_names(lon_min, lat_min, lon_max, lat_max):
    names = []
    for lat in range(math.floor(lat_min), math.floor(lat_max) + 1):
        for lon in range(math.floor(lon_min), math.floor(lon_max) + 1):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            names.append(f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM")
    return names


def fetch_elevation(raster: Raster, crs, cache_dir="data/cache/dem"):
    key = hashlib.sha256(json.dumps([str(crs), raster.x0, raster.y0, raster.res,
                                     raster.nx, raster.ny]).encode()).hexdigest()[:16]
    cache = Path(cache_dir) / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    to_geo = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    bounds = to_geo.transform_bounds(*raster.bounds)
    elevation = np.full(raster.shape, np.nan, dtype=np.float32)
    env = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif"}
    with rasterio.Env(**env):
        for name in tile_names(*bounds):
            try:
                src = rasterio.open("/vsicurl/" + TILE_URL.format(name=name))
            except rasterio.errors.RasterioIOError:
                continue
            with src, WarpedVRT(src, crs=crs, transform=raster.top_down_transform(),
                                width=raster.nx, height=raster.ny,
                                resampling=Resampling.bilinear, nodata=np.nan,
                                dtype="float32") as vrt:
                tile = np.flipud(vrt.read(1))
            elevation = np.where(np.isfinite(tile), tile, elevation)
    if np.isnan(elevation).all():
        raise ValueError("No DEM coverage for this raster")
    elevation = np.where(np.isnan(elevation), 0.0, elevation)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, elevation)
    return elevation
