"""Unpack the compressed patch files (.npz) into uncompressed .npy arrays that DataLoader workers memory-map.

    python scripts/unpack_patches.py [ZIP_YEAR ...]     # default: every data/interim/height/<zip>_<year> folder

For data/interim/height/{zip}_{year}/{i}_{j}.npz writes data/interim/height_npy/{zip}_{year}/{i}_{j}_{img,ndsm,veg}.npy
(img uint8 3x1024x1024, ndsm float32 512x512 with NaN, veg bool). Sparse LiDAR years also get {i}_{j}_ndsm_filled.npy:
the nearest-neighbour filled anchor that bitemporal.Pairs used to compute on every load. Memory-mapped reads share
the OS page cache, so six workers do not hold six copies of the data. Resumable; writes are atomic.
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
CITY = os.environ.get("POPUP_DATA") == "citywide"   # M4: data/interim/height_city -> height_npy_city
SRC = ROOT / ("data/interim/height_city" if CITY else "data/interim/height")
DST = ROOT / ("data/interim/height_npy_city" if CITY else "data/interim/height_npy")


def save(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def unpack(folder):
    out = DST / folder.name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(folder.glob("*.npz")):
        stem = f.stem
        if (out / f"{stem}_veg.npy").exists():
            continue
        d = np.load(f)
        h = d["ndsm"].astype(np.float32)
        save(out / f"{stem}_img.npy", np.ascontiguousarray(d["img"]))
        save(out / f"{stem}_ndsm.npy", h)
        if np.isfinite(h).mean() < 0.8:  # sparse 2010 / 2015 anchor: precompute the nearest-neighbour fill
            idx = ndimage.distance_transform_edt(~np.isfinite(h), return_distances=False, return_indices=True)
            save(out / f"{stem}_ndsm_filled.npy", h[tuple(idx)])
        save(out / f"{stem}_veg.npy", np.ascontiguousarray(d["veg"]))
        n += 1
    return n


if __name__ == "__main__":
    folders = [SRC / a for a in sys.argv[1:]] or sorted(p for p in SRC.iterdir() if p.is_dir())
    for fo in folders:
        print(f"{fo.name}: {unpack(fo)} patches unpacked", flush=True)
    print("UNPACK DONE")
