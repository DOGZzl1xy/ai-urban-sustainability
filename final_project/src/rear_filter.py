"""Stricter rear-addition test (v2), motivated by the manual review: rear-only flags were mostly trees, vehicles and
edge noise. New building cells must
  1. survive a 3x3 morphological opening (removes scattered canopy returns and 1-cell edge slivers),
  2. form a connected blob of >= MIN_BLOB_M2, and
  3. touch the existing 2018 building / 2017 footprint of the same parcel (within ATTACH_M) — a rear addition is
     built onto the house; parked trucks and trees in yards or lots are not.
The parcel needs an existing building in 2018 (cover >= 0.2), otherwise the change is new construction, not an addition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rasterio import features as rfeat
from rasterio.transform import from_origin
from scipy import ndimage as ndi

import change as C
from rasters import CELL, FT

CELL_M2 = (CELL * FT) ** 2
MIN_BLOB_M2 = 4.0
ATTACH_M = 1.0
REAR_M2 = 10.0


def rear_v2(units, footprints, rules=C.Rules(), vehicle_mask=None):
    """``vehicle_mask``: optional bool grid of cells under detected vehicles (v3); those cells are not new building."""
    a, grid = C.load(2018)
    b, _ = C.load(2022)
    veg18 = np.nan_to_num(a["veg"], nan=0) > rules.veg_major
    veg22 = np.nan_to_num(b["veg"], nan=0) > rules.veg_major
    b18 = (a["ndsm"] > rules.bldg_h) & ~veg18
    b22 = (b["ndsm"] > rules.bldg_h) & ~veg22
    shape = (grid.ny, grid.nx)
    fp = rfeat.rasterize(((g, 1) for g in footprints.geometry), out_shape=shape, transform=grid.transform, fill=0,
                         dtype="uint8").astype(bool)
    st = np.ones((3, 3), bool)
    attach = int(np.ceil(ATTACH_M / (CELL * FT)))
    rows = []
    for u, geom in zip(units.unit, units.geometry):
        x0, y0, x1, y1 = geom.bounds
        r0, c0 = grid.index(np.array([x0]), np.array([y1]))
        r1, c1 = grid.index(np.array([x1]), np.array([y0]))
        r0, c0 = max(int(r0[0]) - 2, 0), max(int(c0[0]) - 2, 0)
        r1, c1 = min(int(r1[0]) + 3, grid.ny), min(int(c1[0]) + 3, grid.nx)
        if r1 <= r0 or c1 <= c0:
            rows.append({"unit": u, "rear2_m2": 0.0})
            continue
        tr = from_origin(grid.x0 + c0 * CELL, grid.y1 - r0 * CELL, CELL, CELL)
        inside = rfeat.rasterize([(geom, 1)], out_shape=(r1 - r0, c1 - c0), transform=tr, fill=0, dtype="uint8").astype(bool)
        w18, w22, wfp = b18[r0:r1, c0:c1], b22[r0:r1, c0:c1], fp[r0:r1, c0:c1]
        new = w22 & ~w18 & ~wfp & inside
        if vehicle_mask is not None:
            new &= ~vehicle_mask[r0:r1, c0:c1]
        new = ndi.binary_opening(new, st)
        lab, n = ndi.label(new, structure=st)
        base = ndi.binary_dilation((w18 | wfp) & inside, st, iterations=attach)
        keep = 0
        for k in range(1, n + 1):
            blob = lab == k
            area = blob.sum() * CELL_M2
            if area >= MIN_BLOB_M2 and (blob & base).any():
                keep += area
        rows.append({"unit": u, "rear2_m2": float(keep)})
    return pd.DataFrame(rows)


def apply_v2(f: pd.DataFrame, r2: pd.DataFrame, rules=C.Rules()) -> pd.DataFrame:
    f = f.drop(columns=[c for c in ("rear2_m2",) if c in f]).merge(r2, on="unit", how="left")
    f["chg_rear_v2"] = (f.rear2_m2 >= REAR_M2) & (f.cover18 >= rules.new_before)
    f["chg_any_build_v2"] = f.chg_new | f.chg_popup | f.chg_rear_v2
    return f


def vehicle_mask(chips_dir, grid, buffer_m=0.5):
    """Rasterise detected vehicles (both years) from the chip detections onto the LiDAR grid."""
    import json

    import geopandas as gpd
    from shapely.geometry import Polygon

    idx = json.loads((chips_dir / "index.json").read_text())
    det = json.loads((chips_dir / "vehicles.json").read_text())
    polys = []
    for k, dets in det.items():
        if k not in idx or not dets:
            continue
        (x0, y0, x1, y1), (w, h) = idx[k]["bounds3857"], idx[k]["size"]
        rx, ry = (x1 - x0) / w, (y1 - y0) / h
        polys += [Polygon([(x0 + c * rx, y1 - r * ry) for c, r in d["poly"]]) for d in dets]
    g = gpd.GeoSeries(polys, crs=3857).to_crs(2272).buffer(buffer_m / FT)
    m = rfeat.rasterize(((p, 1) for p in g if not p.is_empty), out_shape=(grid.ny, grid.nx), transform=grid.transform,
                        fill=0, dtype="uint8").astype(bool)
    return m, len(polys)
