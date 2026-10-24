"""Leave-one-snapshot-out check of the interpolation.

Hide snapshot k, interpolate across the 24 h gap from k-1 to k+1, and compare
the predicted burned area at t_k with the hidden observation. Scores cover
only the growth band (cells burned after t_{k-1}), so the unchanged interior
does not inflate them. A physics-free baseline (straight-line distance, uniform
pacing) on the same raster shows what wind, slope and pacing add.
"""

import numpy as np

from .fire import prepare_inputs
from .spread import SpreadModel, interpolate


def _score(predicted, observed, before, after):
    pred, obs = predicted & ~before, observed & ~before
    union = (pred | obs).sum()
    total = max(1, (after & ~before).sum())
    return {"band_iou": float((pred & obs).sum() / union) if union else 1.0,
            "predicted_growth_fraction": float(pred.sum() / total),
            "observed_growth_fraction": float(obs.sum() / total)}


def leave_one_out(sequence, crs, *, model=SpreadModel(noise_sigma=0.0), seed=0, min_growth_cells=50,
                  **input_options):
    """Noise is off by default: it adds run-to-run variety, not accuracy."""
    inputs = prepare_inputs(sequence, crs, **input_options)
    baseline = SpreadModel(noise_sigma=0.0)
    masks = [inputs.raster.mask(p.geometry) for p in sequence]
    rows = []
    for k in range(1, len(sequence) - 1):
        before = masks[k - 1]
        observed = masks[k] | before
        after = masks[k + 1] | observed
        if (after & ~before).sum() < min_growth_cells:
            continue  # nothing meaningful to interpolate
        geoms = [sequence[k - 1].geometry, sequence[k + 1].geometry]
        times = [inputs.times_s[k - 1], inputs.times_s[k + 1]]
        t_k = inputs.times_s[k]
        physics, _ = interpolate(inputs.raster, geoms, times, model=model,
                                 drivers=inputs.drivers([k - 1, k + 1]),
                                 elevation=inputs.elevation, seed=seed)
        plain, _ = interpolate(inputs.raster, geoms, times, model=baseline)
        p, b = _score(physics <= t_k, observed, before, after), _score(plain <= t_k, observed, before, after)
        rows.append({"hidden": k, "label": sequence[k].timestamp.isoformat(),
                     "growth_cells": int((after & ~before).sum()),
                     "physics_iou": p["band_iou"], "baseline_iou": b["band_iou"],
                     "observed_fraction": p["observed_growth_fraction"],
                     "physics_fraction": p["predicted_growth_fraction"],
                     "baseline_fraction": b["predicted_growth_fraction"]})
    return rows


def summarise(rows):
    if not rows:
        return {}
    weights = np.array([r["growth_cells"] for r in rows], dtype=float)
    out = {"intervals": len(rows)}
    for key in ("physics", "baseline"):
        out[f"{key}_iou_weighted"] = float(np.average([r[f"{key}_iou"] for r in rows], weights=weights))
        out[f"{key}_fraction_abs_error"] = float(np.mean([abs(r[f"{key}_fraction"] - r["observed_fraction"]) for r in rows]))
    out["physics_better_count"] = sum(r["physics_iou"] > r["baseline_iou"] for r in rows)
    return out
