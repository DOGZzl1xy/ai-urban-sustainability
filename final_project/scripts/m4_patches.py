"""M4 patch selection on the citywide split (results/citywide/M4_plan.md, splits_citywide.md).

    python scripts/m4_patches.py count                 # patch / pair counts per period and split, steps and run time
    python scripts/m4_patches.py index                 # write data/interim/m4_patches.csv (the selection, no images)

Patches are 512 x 512 cells (256 m) on each ZIP's window of the citywide grid (data/interim/citywide_rasters), as in
Stage 3 (scripts/height_dataset.py), with the same validity rule (share of cells with a LiDAR height >= 0.8 for
2018 / 2022, >= 0.3 for the sparse 2010 / 2015) and >= 50% of the patch inside its ZIP. Added for the citywide study
area: >= 25% of the patch inside the study-area mask. Split (splits_citywide.md): the patch takes the split of the
1 km block of its centre; a patch touching a test ZIP buffered by 100 m is never used; a train patch within 50 m of a
val block is dropped. Test ZIPs are not read.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features as rfeat
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
import tiles as T  # noqa: E402
from rasters import CELL, FT, Grid  # noqa: E402

RAS = ROOT / "data/interim/citywide_rasters"
YEARS = (2010, 2015, 2018, 2022)
PERIODS = ("2010-2015", "2010-2018", "2010-2022", "2015-2018", "2015-2022", "2018-2022")
MIN_INSIDE_ZIP, MIN_INSIDE_MASK = 0.5, 0.25
M3B_PAIRS, M3B_ITERS = 2052, 8000


def select():
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = list(zt[~zt.test].zip)
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
    bl = gpd.read_file(ROOT / "data/interim/splits_citywide.gpkg", layer="blocks")
    split_of = dict(zip(bl.block, bl.split))
    test_buf = T.study_area(tuple(zt[zt.test].zip), buffer_ft=100 / FT).union_all()
    val_buf = bl[bl.split == "val"].union_all().buffer(50 / FT)
    blk = 1000 / FT
    rows = []
    for z in zips:
        zarea = T.study_area((z,), buffer_ft=0).union_all()
        for y in YEARS:
            f = RAS / f"{z}_{y}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            g = Grid(float(d["x0"]), float(d["y1"]), int(d["nx"]), int(d["ny"]))
            nd = d["ndsm"]
            inm = rfeat.rasterize([(zarea.intersection(mask), 1)], out_shape=(g.ny, g.nx), transform=g.transform,
                                  fill=0, dtype="uint8")
            minv = 0.8 if y >= 2018 else 0.3
            for p in H.patches(g.x0, g.y1, g.nx, g.ny):
                sl = np.s_[p.i * H.P:(p.i + 1) * H.P, p.j * H.P:(p.j + 1) * H.P]
                if inm[sl].mean() < MIN_INSIDE_MASK:   # inside ZIP AND mask >= 25% (cheap test first)
                    continue
                valid = float(np.isfinite(nd[sl]).mean())
                b = box(*p.bounds)
                inside = b.intersection(zarea).area / b.area
                if valid < minv or inside < MIN_INSIDE_ZIP:
                    continue
                c = b.centroid
                split = split_of.get(f"{int(c.x // blk)}_{int(c.y // blk)}", "none")
                if b.intersects(test_buf) or split in ("test", "none"):
                    split = "excluded"
                elif split == "train" and b.intersects(val_buf):
                    split = "buffer"
                rows.append(dict(zip=z, year=y, i=p.i, j=p.j, x0=p.x0, y1=p.y1, valid=round(valid, 3),
                                 inside=round(inside, 3), in_mask=round(float(inm[sl].mean()), 3), split=split))
        print(f"  {z}", flush=True)
    return pd.DataFrame(rows)


def pairs(idx):
    out = []
    for per in PERIODS:
        y0, y1 = (int(v) for v in per.split("-"))
        a, b = idx[idx.year == y0], idx[idx.year == y1]
        m = a.merge(b[["zip", "i", "j"]], on=["zip", "i", "j"])
        out.append(m.assign(period=per))
    return pd.concat(out, ignore_index=True)


def steps(n_train):
    iters = max(8000, 500 * round(M3B_ITERS * n_train / M3B_PAIRS / 500))
    return iters, iters // 2, max(500, 500 * round(iters / 16 / 500))


if __name__ == "__main__":
    out = ROOT / "data/interim/m4_patches.csv"
    idx = pd.read_csv(out, dtype={"zip": str}) if out.exists() and sys.argv[1:] == ["count"] else select()
    if sys.argv[1:] == ["index"]:
        idx.to_csv(out, index=False)
    pr = pairs(idx)
    print(pr.groupby(["period", "split"]).size().unstack(fill_value=0).to_string())
    n_tr = int((pr.split == "train").sum())
    n_va = int(((pr.split == "val") & (pr.period == "2018-2022")).sum())
    it, fr, ev = steps(n_tr)
    print(f"train pairs {n_tr} (M3b {M3B_PAIRS}); val pairs 2018-2022 {n_va}; "
          f"iters {it}, frozen {fr}, eval every {ev}")
    for sps in (1.97, 1.5):
        print(f"  at {sps} steps/s: {it / sps / 3600:.1f} h training")
