"""High-rise flag (study_area.md): 100 m cells, 250 m cells and 2019 block groups where the 90th percentile of
building height exceeds 25 m in 2018 or 2025. Flagged areas are left out of the main results and reported separately.

    python scripts/highrise_flag.py [--role trainval|test|all]

Building cell = nDSM > 2.5 m and vegetation returns not the majority (the change rules' definition), counted only
inside the study-area mask and inside the ZIP itself (ZIP windows overlap by 300 ft; each 0.5 m cell is counted once).
Heights are binned at 0.25 m per area and merged across ZIPs, so the p90 of an area split by a ZIP line is exact to
the bin. Cell keys match scripts/grid_eval.py: f"{x // s}_{y // s}" in EPSG:2272 feet, s = 100 m or 250 m.
Output: results/citywide/highrise_{role}.csv (unit, key, n building cells, p90 2018, p90 2025, flagged).
Test ZIPs are only run with --role test / all, at the round-2 final test; nothing about them is printed. --role all
merges train / val and test ZIPs, so cells and block groups that straddle a test ZIP use both parts (study_area.md).
"""

import argparse
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features as rfeat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import change as C  # noqa: E402
import tiles as T  # noqa: E402
from rasters import CELL, FT, Grid  # noqa: E402

RAS = ROOT / "data/interim/citywide_rasters"
LIMIT_M = 25.0
BIN = 0.25
NB = int(120 / BIN)   # heights 0-120 m; taller cells go to the last bin


def block_groups():
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2019_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    bg = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2019_42_bg.zip'}!{name}")
    return bg[bg.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]].reset_index(drop=True)


def zip_hist(zip_code, year, mask, bg):
    z = np.load(RAS / f"{zip_code}_{year}.npz")
    g = Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))
    area = T.study_area((zip_code,), buffer_ft=0).union_all().intersection(mask)
    inside = rfeat.rasterize([(area, 1)], out_shape=(g.ny, g.nx), transform=g.transform, fill=0,
                             dtype="uint8").astype(bool)
    rules = C.Rules()
    nd = z["ndsm"]
    b = inside & (nd > rules.bldg_h) & ~(np.nan_to_num(z["veg"], nan=0) > rules.veg_major)
    r, c = np.nonzero(b)
    h = np.minimum((nd[r, c] / BIN).astype(np.int64), NB - 1)
    x = g.x0 + (c + 0.5) * CELL
    y = g.y1 - (r + 0.5) * CELL
    bgid = rfeat.rasterize(zip(bg.geometry, bg.index + 1), out_shape=(g.ny, g.nx), transform=g.transform, fill=0,
                           dtype="int32")[r, c]
    out = {}
    for unit, key in (("c100", None), ("c250", None), ("bg", bgid)):
        if unit == "bg":
            k = np.where(key > 0, key - 1, -1)
            ok = k >= 0
            df = pd.DataFrame({"key": bg.GEOID.to_numpy()[k[ok]], "h": h[ok]})
        else:
            s = (100 if unit == "c100" else 250) / FT
            df = pd.DataFrame({"key": [f"{a}_{b_}" for a, b_ in zip((x // s).astype(int), (y // s).astype(int))],
                               "h": h})
        out[unit] = df.groupby(["key", "h"]).size()
    return out


def p90(counts):
    """counts: Series indexed by (key, bin) -> 90th percentile height per key (upper edge of the bin)."""
    res = {}
    for key, s in counts.groupby(level=0):
        s = s.droplevel(0).sort_index()
        cum = s.cumsum()
        res[key] = ((cum.index[np.searchsorted(cum.to_numpy(), 0.9 * cum.iloc[-1])] + 1) * BIN, int(cum.iloc[-1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["trainval", "test", "all"], default="trainval")
    a = ap.parse_args()
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = list(zt.zip) if a.role == "all" else list(zt[zt.test == (a.role == "test")].zip)
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
    bg = block_groups()
    acc = {(u, y): [] for u in ("c100", "c250", "bg") for y in (2018, 2025)}
    for i, z in enumerate(zips, 1):
        for y in (2018, 2025):
            for u, s in zip_hist(z, y, mask, bg).items():
                acc[(u, y)].append(s)
        if a.role == "trainval":
            print(f"  {i}/{len(zips)} {z}", flush=True)
    rows = []
    for u in ("c100", "c250", "bg"):
        q = {y: p90(pd.concat(acc[(u, y)]).groupby(level=[0, 1]).sum()) for y in (2018, 2025)}
        for key in sorted(set(q[2018]) | set(q[2025])):
            h18, n18 = q[2018].get(key, (np.nan, 0))
            h25, n25 = q[2025].get(key, (np.nan, 0))
            rows.append(dict(unit=u, key=key, n_bldg_cells_2018=n18, n_bldg_cells_2025=n25, p90_2018=h18,
                             p90_2025=h25, flagged=bool((h18 > LIMIT_M) or (h25 > LIMIT_M))))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / f"results/citywide/highrise_{a.role}.csv", index=False)
    if a.role == "trainval":
        print(out.groupby("unit").agg(areas=("key", "size"), flagged=("flagged", "sum")).to_string())


if __name__ == "__main__":
    main()
