"""Pack upload batch 1 for cloud inference (stage 6, intermediate years): no test-area data, no upload.

    python cloud/pack_batch1.py OUT_DIR

Contents (train / val ZIPs only; everything touching a sealed test ZIP buffered by 100 m is left out or blanked):
  rasters_NN.tar   anchor-year LiDAR layers, data/interim/anchor_rasters/{ZIP}_{YEAR}.npz for 2018 and 2022:
                   ndsm and veg as float32 exactly as in citywide_rasters (lossless) + the grid keys x0, y1, nx, ny,
                   r0_city, c0_city. Cells touching a test ZIP + 100 m are set to NaN in both layers.
  imagery_YEAR.tar aerial tiles data/raw/ortho/tiles/{YEAR}/19_{x}_{y}.jpg for 2018, 2019, 2020, 2022, 2023, 2024,
                   unchanged JPEGs; tiles touching a test ZIP + 100 m are not included.
  units.tar        data/interim/anchor_units/{ZIP}.parquet: unit polygons without address and OPA account columns;
                   units touching a test ZIP + 100 m are dropped.
  MANIFEST.csv (every packed file: archive, path, bytes, sha256), SHA256SUMS (archives), exclusion_check_batch1.md.
Runs at idle CPU and very low I/O priority. Not needed for M4 training (that uses the first package only).
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
OUT = Path(a.out)
TEST = ("19106", "19116", "19127", "19134", "19136", "19139", "19150", "19153")
ANCHOR = (2018, 2022)
IMG_YEARS = (2018, 2019, 2020, 2022, 2023, 2024)
RAS = ROOT / "data/interim/citywide_rasters"
TAR_BYTES = 2 * 1000 ** 3
DROP_COLS = ("address", "opas")


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


low_priority()
t0 = time.time()
OUT.mkdir(parents=True, exist_ok=True)
zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
TV = sorted(zt[~zt.test].zip)
assert not set(TV) & set(TEST) and len(TV) == 38
tb = T.study_area(TEST, buffer_ft=100 / FT).union_all()  # EPSG:2272, test ZIPs + 100 m
man = []

# 1. anchor rasters: ndsm + veg, test buffer blanked (cells touching it), lossless float32
part, tf, size = 0, None, 0
blank = {}
for zp in TV:
    for year in ANCHOR:
        with np.load(RAS / f"{zp}_{year}.npz") as d:
            k = {n: d[n] for n in ("ndsm", "veg", "x0", "y1", "nx", "ny", "r0_city", "c0_city")}
        nd, veg = k["ndsm"], k["veg"]
        assert nd.dtype == np.float32 and veg.dtype == np.float32 and nd.shape == (int(k["ny"]), int(k["nx"]))
        tr = from_origin(float(k["x0"]), float(k["y1"]), CELL, CELL)
        win = box(float(k["x0"]), float(k["y1"]) - nd.shape[0] * CELL, float(k["x0"]) + nd.shape[1] * CELL,
                  float(k["y1"]))
        m = np.zeros(nd.shape, bool)
        if win.intersects(tb):
            m = rfeat.geometry_mask([tb.intersection(win.buffer(10))], nd.shape, tr, invert=True, all_touched=True)
            nd, veg = nd.copy(), veg.copy()
            nd[m], veg[m] = np.nan, np.nan
        blank[(zp, year)] = int(m.sum())
        b = io.BytesIO()
        np.savez_compressed(b, ndsm=nd, veg=veg, **{n: k[n] for n in ("x0", "y1", "nx", "ny", "r0_city", "c0_city")})
        data = b.getvalue()
        if tf is None or size + len(data) > TAR_BYTES:
            if tf:
                tf.close()
            tf, size = tarfile.open(OUT / f"rasters_{part:02d}.tar", "w"), 0
            part += 1
        path = f"data/interim/anchor_rasters/{zp}_{year}.npz"
        add(tf, path, data)
        size += len(data)
        man.append(dict(archive=Path(tf.name).name, path=path, bytes=len(data), sha256=sha(data)))
        print(f"raster {zp}_{year}: {len(data) / 1e9:.3f} GB, blanked {blank[(zp, year)]:,} cells "
              f"({time.time() - t0:.0f}s)", flush=True)
tf.close()

# 2. imagery tiles, without tiles touching the test buffer
tb3857 = gpd.GeoSeries([tb], crs=2272).to_crs(3857).iloc[0]
r = O._res(19) * 256


def tile_box(tx, ty):
    return box(tx * r - O.ORIGIN, O.ORIGIN - (ty + 1) * r, (tx + 1) * r - O.ORIGIN, O.ORIGIN - ty * r)


img_stats = []
for year in IMG_YEARS:
    d = ROOT / f"data/raw/ortho/tiles/{year}"
    names = sorted(e.name for e in os.scandir(d) if e.name.endswith(".jpg"))
    xy = [tuple(map(int, n[:-4].split("_")[1:])) for n in names]
    hit = np.zeros(len(names), bool)
    hit[STRtree([tile_box(*t) for t in xy]).query(tb3857, predicate="intersects")] = True
    keep = [n for n, h in zip(names, hit) if not h]
    with tarfile.open(OUT / f"imagery_{year}.tar", "w") as tf:
        for n in keep:
            data = (d / n).read_bytes()
            path = f"data/raw/ortho/tiles/{year}/{n}"
            add(tf, path, data)
            man.append(dict(archive=f"imagery_{year}.tar", path=path, bytes=len(data), sha256=sha(data)))
    img_stats.append((year, len(names), int(hit.sum()), len(keep)))
    print(f"imagery {year}: {len(keep):,} of {len(names):,} tiles packed, {int(hit.sum()):,} touching the test "
          f"buffer left out ({time.time() - t0:.0f}s)", flush=True)

# 3. unit polygons without address / OPA columns, without units touching the test buffer
unit_stats = []
with tarfile.open(OUT / "units.tar", "w") as tf:
    for zp in TV:
        u = gpd.read_parquet(ROOT / f"data/interim/citywide_units/{zp}.parquet")
        n0 = len(u)
        u = u[~u.intersects(tb)].drop(columns=[c for c in DROP_COLS if c in u.columns])
        b = io.BytesIO()
        u.to_parquet(b)
        data = b.getvalue()
        path = f"data/interim/anchor_units/{zp}.parquet"
        add(tf, path, data)
        man.append(dict(archive="units.tar", path=path, bytes=len(data), sha256=sha(data)))
        unit_stats.append((zp, n0, n0 - len(u), list(u.columns)))
print(f"units: {sum(s[1] - s[2] for s in unit_stats):,} kept, {sum(s[2] for s in unit_stats):,} touching the "
      f"test buffer dropped ({time.time() - t0:.0f}s)", flush=True)

man = pd.DataFrame(man)
man.to_csv(OUT / "MANIFEST.csv", index=False)
archives = sorted(p for p in OUT.glob("*.tar"))
(OUT / "SHA256SUMS").write_text("".join(f"{sha_file(p)}  {p.name}\n" for p in archives), encoding="utf-8")

# 4. exclusion check, re-read from the written archives
bad_rast, bad_tiles, bad_units, cols = 0, 0, 0, set()
for p in archives:
    with tarfile.open(p) as tf:
        for m in tf:
            if p.name.startswith("rasters_"):
                with np.load(io.BytesIO(tf.extractfile(m).read())) as d:
                    nd, veg = d["ndsm"], d["veg"]
                    tr = from_origin(float(d["x0"]), float(d["y1"]), CELL, CELL)
                    win = box(float(d["x0"]), float(d["y1"]) - nd.shape[0] * CELL,
                              float(d["x0"]) + nd.shape[1] * CELL, float(d["y1"]))
                    if win.intersects(tb):
                        mk = rfeat.geometry_mask([tb.intersection(win.buffer(10))], nd.shape, tr, invert=True,
                                                 all_touched=True)
                        bad_rast += int((np.isfinite(nd[mk]) | np.isfinite(veg[mk])).sum())
            elif p.name.startswith("imagery_"):
                tx, ty = map(int, Path(m.name).stem.split("_")[1:])
                bad_tiles += int(tile_box(tx, ty).intersects(tb3857))
            else:
                u = gpd.read_parquet(io.BytesIO(tf.extractfile(m).read()))
                bad_units += int(u.intersects(tb).sum())
                cols |= set(u.columns)
bad_names = man[man.path.str.contains("|".join(TEST))]
addr_cols = sorted(c for c in cols if "addr" in c.lower() or c.lower() in DROP_COLS)
ok = bad_rast == 0 and bad_tiles == 0 and bad_units == 0 and bad_names.empty and not addr_cols
img_rows = "".join(f"| {y} | {n:,} | {h:,} | {k:,} |\n" for y, n, h, k in img_stats)
(OUT / "exclusion_check_batch1.md").write_text(
    f"# Exclusion check of upload batch 1 (inference data)\n\n"
    f"Run {time.strftime('%Y-%m-%d %H:%M')} by `cloud/pack_batch1.py` on the written archives.\n\n"
    f"**Result: {'PASS' if ok else 'FAIL'}**\n\n"
    f"| Check | Result |\n|---|---|\n"
    f"| Anchor raster cells (2018, 2022; 38 train / val ZIP windows) with data inside a test ZIP + 100 m | {bad_rast} |\n"
    f"| Imagery tiles touching a test ZIP + 100 m | {bad_tiles} |\n"
    f"| Units touching a test ZIP + 100 m | {bad_units} |\n"
    f"| Packed file paths naming a test ZIP ({', '.join(TEST)}) | {len(bad_names)} |\n"
    f"| Address or OPA account columns in the unit tables | {len(addr_cols)} |\n\n"
    f"Blanked raster cells (touching the buffer, set to NaN): {sum(blank.values()):,} in "
    f"{sum(v > 0 for v in blank.values())} of {len(blank)} ZIP-year windows. "
    f"Units dropped: {sum(s[2] for s in unit_stats):,} of {sum(s[1] for s in unit_stats):,}.\n\n"
    f"| Imagery year | Tiles in cache | Touching the buffer (left out) | Packed |\n|---|---|---|---|\n{img_rows}\n"
    f"Unit columns kept: {', '.join(unit_stats[0][3])}.\n\n"
    f"Geometry: test ZIP polygons (`data/raw/boundaries/zip_codes.geojson`, EPSG:2272) buffered by 100 m; rasters "
    f"checked cell by cell (all cells touching the buffer), tiles by their zoom-19 footprint in EPSG:3857.\n",
    encoding="utf-8")
print(f"exclusion check {'PASS' if ok else 'FAIL'}; archives {sum(p.stat().st_size for p in archives) / 1e9:.2f} GB; "
      f"total {time.time() - t0:.0f}s", flush=True)
