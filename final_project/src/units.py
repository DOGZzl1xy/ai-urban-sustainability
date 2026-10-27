"""Analysis units for change detection: 2016 PWD parcels, merged with 2025 parcels where lots were split or combined.

Two parcels (one per vintage) are linked when their overlap is >= 30% of either one's area; each connected component of
that graph is one unit. A unit keeps every OPA account from both vintages, so permits filed on the old or the new lot
can be matched.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
LINK_SHARE = 0.30


def _parcels(path, area, vintage):
    p = gpd.read_file(path, bbox=tuple(area.total_bounds)).to_crs(2272)
    p = p[p.representative_point().within(area.union_all())].copy()
    p["opa"] = pd.to_numeric(p.BRT_ID, errors="coerce").astype("Int64").astype(str).str.zfill(9)
    p.loc[p.BRT_ID.isna(), "opa"] = None
    p["vintage"] = vintage
    p["a"] = p.area
    return p[["PARCELID", "opa", "ADDRESS", "BLDG_DESC", "vintage", "a", "geometry"]].reset_index(drop=True)


def build(zip_area: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    area = zip_area[["geometry"]]
    old = _parcels(RAW / "parcels/PhillyWater_PWD_PARCELS/PhillyWater_PWD_PARCELS.shp", area, 2016)
    new = _parcels(RAW / "parcels/PhillyWater_PWD_PARCELS2025/PhillyWater_PWD_PARCELS2025.shp", area, 2025)
    allp = pd.concat([old, new], ignore_index=True)
    allp = gpd.GeoDataFrame(allp, geometry="geometry", crs=2272)
    n_old = len(old)
    ov = gpd.overlay(old.assign(i=np.arange(n_old))[["i", "a", "geometry"]],
                     new.assign(j=np.arange(len(new)) + n_old, b=new.a)[["j", "b", "geometry"]],
                     how="intersection", keep_geom_type=True)
    ov["share"] = ov.area / np.minimum(ov.a, ov.b)
    ov = ov[ov.share >= LINK_SHARE]
    n = len(allp)
    g = coo_matrix((np.ones(len(ov)), (ov.i, ov.j)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    allp["comp"] = comp
    # unit geometry = union of the 2016 parcels in the component (2025 ones if the component has no 2016 parcel)
    base = allp[allp.vintage == 2016]
    extra = allp[~allp.comp.isin(base.comp) & (allp.vintage == 2025)]
    geo = pd.concat([base, extra]).dissolve("comp").geometry
    agg = allp.groupby("comp").agg(
        opas=("opa", lambda s: sorted({x for x in s if isinstance(x, str) and x.isdigit()})),
        n_2016=("vintage", lambda s: int((s == 2016).sum())),
        n_2025=("vintage", lambda s: int((s == 2025).sum())),
        address=("ADDRESS", "first"),
        bldg_desc_2016=("BLDG_DESC", lambda s: s.iloc[0]),
    )
    units = gpd.GeoDataFrame(agg.join(geo), geometry="geometry", crs=2272)
    units["area_m2"] = units.area * 0.3048 ** 2
    units["merged_or_split"] = (units.n_2016 != 1) | (units.n_2025 != 1)
    units["is_row"] = units.bldg_desc_2016.fillna("").str.contains(r"\bROW\b")
    units.index.name = "unit"
    return units.reset_index()


if __name__ == "__main__":
    import sys

    import tiles as T

    zc = sys.argv[1] if len(sys.argv) > 1 else "19146"   # python src/units.py [ZIP] -> data/interim/units_{ZIP}.parquet
    z = T.study_area((zc,), buffer_ft=0)
    u = build(z)
    out = ROOT / "data" / "interim"
    out.mkdir(parents=True, exist_ok=True)
    u.to_parquet(out / f"units_{zc}.parquet")
    print(f"{len(u):,} units | merged/split {u.merged_or_split.sum():,} | 2016-only {(u.n_2025 == 0).sum()} | "
          f"2025-only {(u.n_2016 == 0).sum()} | rowhouse {u.is_row.sum():,}")
    print(u.area_m2.describe().round(1))
    fp = gpd.read_file(RAW / "footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp", bbox=tuple(z.total_bounds))
    print(f"2017 footprints in {zc} bbox: {len(fp):,}")
