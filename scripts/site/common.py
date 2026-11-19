"""Shared settings for the site exports: data paths and the km grid centred on the early fire."""
import functools
import json
import os

GPKG, CACHE, FIRE = "data/raw/LargeFires_2018.gpkg", "data/cache", 3885
_CENTRE = os.path.join(CACHE, "site_centre.json")


@functools.lru_cache(None)
def centre():
    if os.path.exists(_CENTRE):
        return tuple(json.load(open(_CENTRE)))
    import shapely
    from fire_sim.fire import load_fire
    _, sequence = load_fire(GPKG, fire_id=FIRE, year=2018, limit=8, seed=0, cache_dir=CACHE)
    c = shapely.union_all([p.geometry for p in sequence[:3]]).centroid
    json.dump([c.x, c.y], open(_CENTRE, "w"))
    return c.x, c.y


def km(x, y):
    cx, cy = centre()
    return [round((x - cx) / 1000, 3), round((y - cy) / 1000, 3)]
