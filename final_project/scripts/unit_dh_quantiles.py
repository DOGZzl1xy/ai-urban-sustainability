"""Extra per-unit height-change statistics for the train / val ZIPs and the six training periods.

    python scripts/unit_dh_quantiles.py OUT.parquet

Same cells as src/change.unit_features: cells of the unit (inset 0.25 m) that are building in y0 (nDSM > 2.5 m,
not majority vegetation), non-vegetation in both years and have a height in both years; dh = nDSM(y1) - nDSM(y0).
Per unit: n_dh_cells, dh_p50, dh_p75, dh_p90 (linear quantiles, as np.median), share_up_1p5 = share of those cells
with dh >= 1.5 m, share_up_2p5 = same for 2.5 m. dh_p50 and share_up_2p5 must equal the existing dh_p50 and frac_up
of data/interim/lidar_change_city (checked; the script stops if they differ). Reads the citywide rasters; one ZIP
and period at a time, started only when enough memory is free; idle CPU and very low I/O priority.
"""

import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from rasterio import features as rfeat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
from rasters import Grid  # noqa: E402

RAS = ROOT / "data/interim/citywide_rasters"
CHG = ROOT / "data/interim/lidar_change_city"
UNITS = ROOT / "data/interim/citywide_units"
PERIODS = ("2010-2015", "2010-2018", "2010-2022", "2015-2018", "2015-2022", "2018-2022")
BYTES_PER_CELL = 18   # two nDSM (float32) + one vegetation layer being read + two boolean masks + unit ids


def low_priority():
    p = psutil.Process()
    p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    try:
        p.ionice(psutil.IOPRIO_VERYLOW if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    except Exception:  # noqa: BLE001
        pass


def epoch(z, y):
    d = np.load(RAS / f"{z}_{y}.npz")
    nd = d["ndsm"]
    veg = np.nan_to_num(d["veg"], nan=0) > C.Rules().veg_major
    return nd, veg, Grid(float(d["x0"]), float(d["y1"]), int(d["nx"]), int(d["ny"]))


def one(z, y0, y1, uid):
    r = C.Rules()
    n0, v0, _ = epoch(z, y0)
    b0 = (n0 > r.bldg_h) & ~v0
    n1, v1, _ = epoch(z, y1)
    sel = (uid > 0) & b0 & ~v0 & ~v1 & np.isfinite(n0) & np.isfinite(n1)
    del b0, v0, v1
    u = uid[sel] - 1
    dh = (n1[sel] - n0[sel]).astype(np.float64)
    del n0, n1, sel
    df = pd.DataFrame({"unit": u, "dh": dh, "up15": dh >= 1.5, "up25": dh >= r.up_h})
    g = df.groupby("unit")
    q = g.dh.quantile([0.5, 0.75, 0.9]).unstack()
    out = pd.DataFrame({"n_dh_cells": g.size(), "dh_p50": q[0.5], "dh_p75": q[0.75], "dh_p90": q[0.9],
                        "share_up_1p5": g.up15.mean(), "share_up_2p5": g.up25.mean()})
    return out.reset_index()


def main():
    low_priority()
    out_path = Path(sys.argv[1])
    idx = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = sorted(zt[~zt.test].zip)
    assert not set(zips) & set(zt[zt.test].zip)
    cells = {}
    for z in zips:
        d = np.load(RAS / f"{z}_2018.npz")
        cells[z] = int(d["nx"]) * int(d["ny"])
    parts, t0 = [], time.time()
    for z in sorted(zips, key=cells.get):   # small windows first; large ones wait for free memory
        need = cells[z] * BYTES_PER_CELL / 1e9 + 1.0
        while psutil.virtual_memory().available / 1e9 < need:
            time.sleep(30)
        units = gpd.read_parquet(UNITS / f"{z}.parquet")[["unit", "geometry"]]
        d = np.load(RAS / f"{z}_2018.npz")
        grid = Grid(float(d["x0"]), float(d["y1"]), int(d["nx"]), int(d["ny"]))
        geoms = units.geometry.buffer(-C.INSET_FT)
        ok = ~geoms.is_empty
        uid = rfeat.rasterize(zip(geoms[ok], units.unit[ok] + 1), out_shape=(grid.ny, grid.nx),
                              transform=grid.transform, fill=0, dtype="int32")
        for per in PERIODS:
            y0, y1 = (int(v) for v in per.split("-"))
            res = one(z, y0, y1, uid)
            ref = pd.read_parquet(CHG / f"{z}_{y0}_{y1}.parquet")[["unit", "n_dh_cells", "dh_p50", "frac_up"]]
            m = ref.merge(res, on="unit", how="left", suffixes=("_ref", ""))
            m["n_dh_cells"] = m.n_dh_cells.fillna(0)
            has = m.n_dh_cells_ref.fillna(0) > 0
            assert (m.n_dh_cells_ref.fillna(0) == m.n_dh_cells).all(), (z, per, "n_dh_cells")
            assert np.allclose(m.dh_p50_ref[has], m.dh_p50[has], atol=1e-5), (z, per, "dh_p50")
            assert np.allclose(m.frac_up[has], m.share_up_2p5[has], atol=1e-9), (z, per, "share >= 2.5 m")
            parts.append(m.drop(columns=["n_dh_cells_ref", "dh_p50_ref", "frac_up"]).assign(zip=z, period=per))
        del uid
        print(f"  {z} ({cells[z] / 1e6:.0f} M cells) {time.time() - t0:.0f}s", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out["n_dh_cells"] = out.n_dh_cells.astype(np.int64)
    out = out[["zip", "period", "unit", "n_dh_cells", "dh_p50", "dh_p75", "dh_p90", "share_up_1p5",
               "share_up_2p5"]]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"DONE {len(out):,} rows, {out.zip.nunique()} ZIPs; checks vs existing dh_p50 / frac_up passed", flush=True)


if __name__ == "__main__":
    main()
