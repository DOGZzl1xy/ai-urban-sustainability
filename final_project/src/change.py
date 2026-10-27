"""Per-unit LiDAR change features (2018 -> 2022) and rule-based change types.

Cells are 0.5 m (0.25 m2). A cell is "building" in an epoch when nDSM > 2.5 m and vegetation returns are not the
majority. Height change is only read on cells that are non-vegetation in both epochs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features as rfeat

from rasters import CELL, FT, Grid

ROOT = Path(__file__).resolve().parents[1]
CELL_M2 = (CELL * FT) ** 2
INSET_FT = 0.25 / FT


@dataclass
class Rules:
    """Thresholds from FEASIBILITY_TEST.md §2 (may be revised once on the recall set; see summary.md)."""
    bldg_h: float = 2.5          # m, nDSM height for a building cell
    veg_major: float = 0.5       # vegetation share above which a cell is masked
    new_before: float = 0.20     # new: building cover < this in 2018 ...
    new_after: float = 0.50      # ... and > this in 2022 (demolition: the reverse)
    up_h: float = 2.5            # pop-up: cells rising >= this much ...
    up_share: float = 0.30       # ... make up >= this share of the 2018 building cells
    rear_m2: float = 10.0        # rear addition: >= this many m2 of new building cells outside the 2017 footprint


def load(year):
    z = np.load(ROOT / f"data/interim/rasters_{year}.npz")
    return {k: z[k] for k in ("ndsm", "veg", "ntot")}, Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))


def unit_features(units: gpd.GeoDataFrame, footprints: gpd.GeoDataFrame, rules: Rules = Rules(),
                  epochs=None, sparse=False) -> pd.DataFrame:
    """epochs = (before, after, grid) with dicts of ndsm / veg / ntot rasters; default: LiDAR 2018 and 2022.
    The height model's predictions are scored by passing predicted rasters here (same rules, same units).
    sparse=True (2015 / 2010 LiDAR, 1-3 pts/m2): building cover is the share of cells WITH data that are building,
    instead of the share of all cells, because most 0.5 m cells are empty."""
    if epochs is None:
        a, grid = load(2018)
        b, grid2 = load(2022)
        assert (grid.nx, grid.ny) == (grid2.nx, grid2.ny)
    else:
        a, b, grid = epochs
    shape = (grid.ny, grid.nx)
    geoms = units.geometry.buffer(-INSET_FT)
    ok = ~geoms.is_empty
    uid = rfeat.rasterize(zip(geoms[ok], units.unit[ok] + 1), out_shape=shape, transform=grid.transform, fill=0,
                          dtype="int32")
    fp = rfeat.rasterize(((g, 1) for g in footprints.geometry), out_shape=shape, transform=grid.transform, fill=0,
                         dtype="uint8").astype(bool)

    veg18 = np.nan_to_num(a["veg"], nan=0) > rules.veg_major
    veg22 = np.nan_to_num(b["veg"], nan=0) > rules.veg_major
    n18, n22 = a["ndsm"], b["ndsm"]
    b18 = (n18 > rules.bldg_h) & ~veg18
    b22 = (n22 > rules.bldg_h) & ~veg22
    both = np.isfinite(n18) & np.isfinite(n22) & ~veg18 & ~veg22
    dh = np.where(both, n22 - n18, np.nan)

    m = uid > 0
    ids = uid[m] - 1
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    starts = np.searchsorted(ids, units.unit.to_numpy())
    ends = np.searchsorted(ids, units.unit.to_numpy(), side="right")
    cols = {k: v[m][order] for k, v in dict(n18=n18, n22=n22, b18=b18, b22=b22, dh=dh, fp=fp, veg18=veg18,
                                                 veg22=veg22, p18=a["ntot"] > 0, p22=b["ntot"] > 0).items()}
    rows = []
    for u, s, e in zip(units.unit, starts, ends):
        c = {k: v[s:e] for k, v in cols.items()}
        n = e - s
        r = {"unit": u, "n_cells": n}
        if n == 0:
            rows.append(r)
            continue
        if sparse:
            f18, f22 = np.isfinite(c["n18"]), np.isfinite(c["n22"])
            r["cover18"] = c["b18"].sum() / max(f18.sum(), 1) if f18.sum() >= 4 else np.nan
            r["cover22"] = c["b22"].sum() / max(f22.sum(), 1) if f22.sum() >= 4 else np.nan
        else:
            r["cover18"], r["cover22"] = c["b18"].mean(), c["b22"].mean()
        r["bldg_m2_18"], r["bldg_m2_22"] = c["b18"].sum() * CELL_M2, c["b22"].sum() * CELL_M2
        for yr in ("18", "22"):
            hb = c["n" + yr][c["b" + yr]]
            r[f"h50_{yr}"] = float(np.median(hb)) if len(hb) else np.nan
            r[f"h90_{yr}"] = float(np.quantile(hb, 0.9)) if len(hb) else np.nan
        dh_b = c["dh"][c["b18"] & np.isfinite(c["dh"])]
        r["n_dh_cells"] = len(dh_b)
        r["frac_up"] = float((dh_b >= rules.up_h).mean()) if len(dh_b) else np.nan
        r["frac_down"] = float((dh_b <= -rules.up_h).mean()) if len(dh_b) else np.nan
        r["dh_p50"] = float(np.median(dh_b)) if len(dh_b) else np.nan
        r["new_outside_fp_m2"] = float((c["b22"] & ~c["b18"] & ~c["fp"]).sum() * CELL_M2)
        r["veg_share_18"], r["veg_share_22"] = c["veg18"].mean(), c["veg22"].mean()
        r["data_18"], r["data_22"] = c["p18"].mean(), c["p22"].mean()
        rows.append(r)
    f = pd.DataFrame(rows)
    return apply_rules(f, rules)


def apply_rules(f: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    f = f.copy()
    f["chg_new"] = (f.cover18 < rules.new_before) & (f.cover22 > rules.new_after)
    f["chg_demo"] = (f.cover18 > rules.new_after) & (f.cover22 < rules.new_before)
    f["chg_popup"] = (f.cover18 >= rules.new_before) & (f.frac_up >= rules.up_share)
    f["chg_rear"] = f.new_outside_fp_m2 >= rules.rear_m2
    f["chg_any_build"] = f.chg_new | f.chg_popup | f.chg_rear
    f["rules"] = str(asdict(rules))
    return f
