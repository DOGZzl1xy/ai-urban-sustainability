"""Pack upload batch 2 for cloud inference (stage 6): the test-area data left out of batch 1. No upload.

    python cloud/pack_batch2.py OUT_DIR

Only after the round-2 test (results/citywide/m4/round2_test_log.csv must exist). With batch 1 this gives the whole
study area. Contents:
  rasters_test_NN.tar  data/interim/anchor_rasters/{ZIP}_{YEAR}.npz for the eight test ZIPs, 2018 and 2022: ndsm and
                       veg float32 as in citywide_rasters (lossless) + grid keys (the batch-1 format).
  restore.tar          data/interim/anchor_restore/{ZIP}_{YEAR}.npz for the train / val windows blanked in batch 1:
                       the original ndsm / veg over the bounding box of the blanked cells (r0, c0 = its top-left
                       cell). cloud/restore_batch2.py pastes them back, which gives the original arrays exactly.
  imagery_YEAR.tar     the tiles of 2018, 2019, 2020, 2022, 2023, 2024 that batch 1 left out (touching a test ZIP
                       + 100 m), unchanged JPEGs.
  units.tar            data/interim/anchor_units/{ZIP}.parquet for all 46 ZIPs, all units (replaces batch 1's
                       tables), without address and OPA account columns.
  MANIFEST.csv (archive, path, bytes, sha256), SHA256SUMS (archives). Idle CPU and very low I/O priority.
"""

import argparse
import hashlib
import io
import os
import sys
import tarfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import psutil
from rasterio import features as rfeat
from rasterio.transform import from_origin
from shapely import STRtree, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ortho_chips as O  # noqa: E402
import tiles as T  # noqa: E402
from rasters import CELL, FT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
a = ap.parse_args()
assert (ROOT / "results/citywide/m4/round2_test_log.csv").exists(), "batch 2 holds test data: only after round 2"
OUT = Path(a.out)
ANCHOR = (2018, 2022)
IMG_YEARS = (2018, 2019, 2020, 2022, 2023, 2024)
RAS = ROOT / "data/interim/citywide_rasters"
TAR_BYTES = 2 * 1000 ** 3
DROP_COLS = ("address", "opas")
GRID_KEYS = ("x0", "y1", "nx", "ny", "r0_city", "c0_city")


def low_priority():
    p = psutil.Process()
    p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    try:
        p.ionice(psutil.IOPRIO_VERYLOW if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    except Exception:  # noqa: BLE001
        pass


def sha(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def add(tf, arcname, data):
    ti = tarfile.TarInfo(arcname)
    ti.size, ti.mtime = len(data), int(time.time())
    tf.addfile(ti, io.BytesIO(data))


def npz(**arrs):
    b = io.BytesIO()
    np.savez_compressed(b, **arrs)
    return b.getvalue()


low_priority()
t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
TEST = tuple(sorted(zt[zt.test].zip))
TV = sorted(zt[~zt.test].zip)
tb = T.study_area(TEST, buffer_ft=100 / FT).union_all()  # the batch-1 exclusion zone
man = []

# 1. anchor rasters of the test ZIPs (whole windows)
part, tf, size = 0, None, 0
for z in TEST:
    for y in ANCHOR:
        with np.load(RAS / f"{z}_{y}.npz") as d:
            data = npz(ndsm=d["ndsm"], veg=d["veg"], **{k: d[k] for k in GRID_KEYS})
        if tf is None or size + len(data) > TAR_BYTES:
            if tf:
                tf.close()
            tf, size = tarfile.open(OUT / f"rasters_test_{part:02d}.tar", "w"), 0
            part += 1
        path = f"data/interim/anchor_rasters/{z}_{y}.npz"
        add(tf, path, data)
        size += len(data)
        man.append(dict(archive=Path(tf.name).name, path=path, bytes=len(data), sha256=sha(data)))
        print(f"test raster {z}_{y}: {len(data) / 1e9:.3f} GB ({time.time() - t0:.0f}s)", flush=True)
tf.close()

# 2. restore patches for the train / val windows blanked in batch 1 (same mask as cloud/pack_batch1.py)
with tarfile.open(OUT / "restore.tar", "w") as tf:
    for z in TV:
        for y in ANCHOR:
            with np.load(RAS / f"{z}_{y}.npz") as d:
                nd, veg = d["ndsm"], d["veg"]
                x0, y1 = float(d["x0"]), float(d["y1"])
            win = box(x0, y1 - nd.shape[0] * CELL, x0 + nd.shape[1] * CELL, y1)
            if not win.intersects(tb):
                continue
            m = rfeat.geometry_mask([tb.intersection(win.buffer(10))], nd.shape, from_origin(x0, y1, CELL, CELL),
                                    invert=True, all_touched=True)
            if not m.any():
                continue
            rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
            r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
            data = npz(ndsm=nd[r0:r1, c0:c1], veg=veg[r0:r1, c0:c1], r0=np.int64(r0), c0=np.int64(c0),
                       shape=np.array(nd.shape))
            path = f"data/interim/anchor_restore/{z}_{y}.npz"
            add(tf, path, data)
            man.append(dict(archive="restore.tar", path=path, bytes=len(data), sha256=sha(data)))
            print(f"restore {z}_{y}: box {r1 - r0} x {c1 - c0}, {len(data) / 1e9:.3f} GB ({time.time() - t0:.0f}s)",
                  flush=True)

# 3. imagery tiles left out of batch 1
tb3857 = gpd.GeoSeries([tb], crs=2272).to_crs(3857).iloc[0]
r = O._res(19) * 256
for year in IMG_YEARS:
    d = ROOT / f"data/raw/ortho/tiles/{year}"
    names = sorted(e.name for e in os.scandir(d) if e.name.endswith(".jpg"))
    xy = [tuple(map(int, n[:-4].split("_")[1:])) for n in names]
    boxes = [box(tx * r - O.ORIGIN, O.ORIGIN - (ty + 1) * r, (tx + 1) * r - O.ORIGIN, O.ORIGIN - ty * r)
             for tx, ty in xy]
    hit = np.zeros(len(names), bool)
    hit[STRtree(boxes).query(tb3857, predicate="intersects")] = True
    with tarfile.open(OUT / f"imagery_{year}.tar", "w") as tf:
        for n in (n for n, h in zip(names, hit) if h):
            data = (d / n).read_bytes()
            path = f"data/raw/ortho/tiles/{year}/{n}"
            add(tf, path, data)
            man.append(dict(archive=f"imagery_{year}.tar", path=path, bytes=len(data), sha256=sha(data)))
    print(f"imagery {year}: {int(hit.sum()):,} tiles ({time.time() - t0:.0f}s)", flush=True)

# 4. all unit tables, no address / OPA columns
with tarfile.open(OUT / "units.tar", "w") as tf:
    for z in TV + list(TEST):
        u = gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")
        u = u.drop(columns=[c for c in DROP_COLS if c in u.columns])
        assert not [c for c in u.columns if "addr" in c.lower()]
        b = io.BytesIO()
        u.to_parquet(b)
        data = b.getvalue()
        path = f"data/interim/anchor_units/{z}.parquet"
        add(tf, path, data)
        man.append(dict(archive="units.tar", path=path, bytes=len(data), sha256=sha(data)))
print(f"units: {len(TV) + len(TEST)} ZIPs ({time.time() - t0:.0f}s)", flush=True)

pd.DataFrame(man).to_csv(OUT / "MANIFEST.csv", index=False)
archives = sorted(OUT.glob("*.tar"))
(OUT / "SHA256SUMS").write_text("".join(f"{sha_file(p)}  {p.name}\n" for p in archives), encoding="utf-8")
print(f"BATCH 2 DONE: {len(archives)} archives, {sum(p.stat().st_size for p in archives) / 1e9:.2f} GB; "
      f"{time.time() - t0:.0f}s", flush=True)
