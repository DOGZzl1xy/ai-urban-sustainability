"""Cut paired image / nDSM patches for one ZIP and one year, and measure image-LiDAR misregistration.

    python scripts/height_dataset.py ZIP YEAR [IMAGE_YEAR]

IMAGE_YEAR defaults to YEAR (same-season imagery; both LiDAR and imagery are spring flights).
Writes data/interim/height/{ZIP}_{YEAR}/{i}_{j}.npz (img uint8 3x2Px2P at 0.25 m, ndsm float16 PxP at 0.5 m, veg bool) and index.csv with the
patch bounds, valid share and the edge-correlation offset. Resumable: existing patches are skipped.
"""

import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
import tiles as T  # noqa: E402

ZIP, YEAR = sys.argv[1], int(sys.argv[2])
IMG_YEAR = int(sys.argv[3]) if len(sys.argv) > 3 else YEAR
MIN_VALID = 0.8 if YEAR >= 2018 else 0.3  # share of cells with a LiDAR height (2010/2015 are sparse)
MIN_INSIDE = 0.5     # share of the patch inside the ZIP (no buffer)

z = np.load(H.raster_path(ZIP, YEAR))
ndsm, veg = z["ndsm"], z["veg"]
area = T.study_area((ZIP,), buffer_ft=0).union_all()
out = H.OUT / f"{ZIP}_{YEAR}" if IMG_YEAR == YEAR else H.OUT / f"{ZIP}_{YEAR}_img{IMG_YEAR}"
out.mkdir(parents=True, exist_ok=True)
todo = []
for p in H.patches(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"])):
    sl = np.s_[p.i * H.P:(p.i + 1) * H.P, p.j * H.P:(p.j + 1) * H.P]
    valid = np.isfinite(ndsm[sl]).mean()
    inside = box(*p.bounds).intersection(area).area / box(*p.bounds).area
    if valid >= MIN_VALID and inside >= MIN_INSIDE:
        todo.append((p, sl, valid, inside))
print(f"{ZIP} {YEAR} (imagery {IMG_YEAR}): {len(todo)} patches", flush=True)


def one(args):
    p, sl, valid, inside = args
    f = out / f"{p.i}_{p.j}.npz"
    if f.exists():
        d = np.load(f)
        img, h = d["img"], d["ndsm"].astype(np.float32)
    else:
        img = H.image_for(p, IMG_YEAR, out_res_ft=H.CELL / 2, size=2 * H.P)  # 0.25 m: room for 0.3 / 0.5 m inputs
        h = ndsm[sl]
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, img=img, ndsm=h.astype(np.float16), veg=veg[sl] > 0.5)
        os.replace(tmp, f)  # never leave a half-written patch behind
    blank = (img == 230).all(0).mean() + (img == 0).all(0).mean()
    half = img.reshape(3, H.P, 2, H.P, 2).mean((2, 4))  # image at the 0.5 m label grid
    (dr, dc), score = H.edge_offset(half, h)
    return {"zip": ZIP, "year": YEAR, "img_year": IMG_YEAR, "i": p.i, "j": p.j, "x0": p.x0, "y1": p.y1,
            "valid": round(valid, 3), "inside": round(inside, 3), "blank": round(float(blank), 3),
            "off_r": dr, "off_c": dc, "off_m": round(float(np.hypot(dr, dc)) * 0.5, 2), "corr": round(score, 3)}


rows = []
with ThreadPoolExecutor(8) as ex:
    for k, r in enumerate(ex.map(one, todo), 1):
        rows.append(r)
        if k % 25 == 0:
            print(f"  {k}/{len(todo)}", flush=True)
with open(out / "index.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
off = np.array([r["off_m"] for r in rows])
print(f"done: {len(rows)} patches; misregistration median {np.median(off):.2f} m, "
      f"share > 1 m: {(off > 1).mean():.1%}")
