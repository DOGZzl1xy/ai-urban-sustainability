"""Cut the M4 train / val patches selected by scripts/m4_patches.py (data/interim/m4_patches.csv).

    POPUP_DATA=citywide python scripts/m4_cut.py [--threads 8]      # cut patches
    POPUP_DATA=citywide python scripts/m4_cut.py --prepare           # real-change masks and unit-id rasters

Same patch format and index.csv as scripts/height_dataset.py (image of the same year as the LiDAR, 0.25 m; nDSM and
vegetation at 0.5 m), written to data/interim/height_city/{ZIP}_{YEAR}/. Resumable: existing patches are reused.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"


def cut(zip_code, year, sel, threads):
    z = np.load(H.raster_path(zip_code, year))
    ndsm, veg = z["ndsm"], z["veg"]
    out = H.OUT / f"{zip_code}_{year}"
    out.mkdir(parents=True, exist_ok=True)

    def one(r):
        p = H.Patch(int(r.i), int(r.j), float(r.x0), float(r.y1))
        sl = np.s_[p.i * H.P:(p.i + 1) * H.P, p.j * H.P:(p.j + 1) * H.P]
        f = out / f"{p.i}_{p.j}.npz"
        if f.exists():
            d = np.load(f)
            img, h = d["img"], d["ndsm"].astype(np.float32)
        else:
            img = H.image_for(p, year, out_res_ft=H.CELL / 2, size=2 * H.P)
            h = ndsm[sl]
            tmp = f.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, img=img, ndsm=h.astype(np.float16), veg=veg[sl] > 0.5)
            os.replace(tmp, f)
        blank = (img == 230).all(0).mean() + (img == 0).all(0).mean()
        half = img.reshape(3, H.P, 2, H.P, 2).mean((2, 4))
        (dr, dc), score = H.edge_offset(half, h)
        return {"zip": zip_code, "year": year, "img_year": year, "i": p.i, "j": p.j, "x0": p.x0, "y1": p.y1,
                "valid": r.valid, "inside": r.inside, "blank": round(float(blank), 3), "off_r": dr, "off_c": dc,
                "off_m": round(float(np.hypot(dr, dc)) * 0.5, 2), "corr": round(score, 3)}

    with ThreadPoolExecutor(threads) as ex:
        rows = list(ex.map(one, sel.itertuples()))
    if not rows:
        return rows
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def prepare_all():
    """What scripts/unitcls.py prepare() builds at the start of training, done ahead of time."""
    import bitemporal as B
    import unitcls as U

    idx = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
    zips = sorted(idx[idx.split.isin(["train", "val"])].zip.unique())
    for per in ("2010-2015", "2010-2018", "2010-2022", "2015-2018", "2015-2022", "2018-2022"):
        y0, y1 = (int(v) for v in per.split("-"))
        pi = B.build_masks(tuple(zips), y0, y1)
        print(f"  masks {per}: {len(pi)} pairs, {int((pi.n_real > 0).sum())} with real change", flush=True)
    for z in zips:
        for yr in (2010, 2015, 2018, 2022):
            U.unit_id_patches(z, yr)
    print("PREPARE DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--prepare", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare_all()
        return
    idx = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
    idx = idx[idx.split.isin(["train", "val"])]
    t0, done = time.time(), 0
    groups = list(idx.groupby(["zip", "year"]))
    for k, ((z, y), sel) in enumerate(groups, 1):
        rows = cut(z, int(y), sel, a.threads)
        done += len(rows)
        off = np.array([r["off_m"] for r in rows])
        blank = np.array([r["blank"] for r in rows])
        print(f"  {k}/{len(groups)} {z} {y}: {len(rows)} patches, misregistration median {np.median(off):.2f} m, "
              f"blank >= 5%: {(blank >= 0.05).sum()} | {done}/{len(idx)} in {time.time() - t0:.0f}s", flush=True)
    print("CUT DONE", flush=True)


if __name__ == "__main__":
    main()
