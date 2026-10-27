"""Streaming LiDAR processing for the citywide study area: download a tile -> rasterise on the citywide 0.5 m grid ->
check -> delete the raw file. Raw files that were on disk before streaming started are never deleted.

    python scripts/stream_lidar.py YEAR [--tiles NAME ...] [--limit N] [--workers 2] [--area all|trainval]

Per tile: data/interim/citywide_tiles/{YEAR}/{name}.npz with the tile's block of the citywide grid (r0, c0, dsm max
of non-vegetation returns, ground min on the 1.5 m grid, vegetation and total return counts) - the same quantities
rasters.tile_rasters computes for the per-ZIP grids, so rasters.Mosaic can assemble any area later.
Log: results/citywide/processing/stream_{YEAR}.csv (tile, MB, download s, process s, points, cells). Pauses while free disk
space is below 50 GB. Runs at below-normal priority. Resumable: finished tiles are skipped.
"""

import argparse
import csv
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rasters as R  # noqa: E402
import tiles as T  # noqa: E402

MIN_FREE_GB = 50
VEG_CLASSES = {2025: (3, 4, 5)}   # classified flight: canopy by class as well as by return structure
OUT = ROOT / "data/interim/citywide_tiles"
KEEP_LIST = T.LIDAR / "keep_raw_files.txt"   # raw files present before streaming started: never deleted


def city_grid():
    """Citywide 0.5 m grid in EPSG:2272, origin on a multiple of the cell size (fixed for the whole project)."""
    m = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").total_bounds
    pad = 500.0
    x0 = np.floor((m[0] - pad) / R.CELL) * R.CELL
    y1 = np.ceil((m[3] + pad) / R.CELL) * R.CELL
    nx = int(np.ceil((m[2] + pad - x0) / R.CELL))
    ny = int(np.ceil((y1 - (m[1] - pad)) / R.CELL))
    return R.Grid(float(x0), float(y1), nx, ny)


def needed(year, area):
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
    if area == "trainval":
        bl = gpd.read_file(ROOT / "data/interim/splits_citywide.gpkg", layer="blocks")
        mask = bl[bl.split != "test"].union_all().intersection(mask)
    t = {2010: T.tiles_2010, 2015: T.tiles_2015, 2018: T.tiles_2018, 2022: T.tiles_2022, 2025: T.tiles_2025}[year]()
    return t[t.intersects(mask)].copy()


def download(row):
    import requests

    f = Path(row.file)
    f.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (f.exists() and f.stat().st_size == row.bytes):
        for attempt in range(6):
            try:
                pos = f.stat().st_size if f.exists() else 0
                hdr = {"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={pos}-"} if pos else {})}
                with requests.get(row.url, headers=hdr, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(f, "ab" if pos and r.status_code == 206 else "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                if f.stat().st_size == row.bytes:
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  retry {row.name}: {e!r}", flush=True)
                time.sleep(10 * (attempt + 1))
    ok = f.exists() and f.stat().st_size == row.bytes
    return row, ok, time.time() - t0


def process(path, out, grid_args, veg_classes=()):
    """Rasterise one tile onto the citywide grid and save; returns (points, cells, seconds)."""
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass
    t0 = time.time()
    grid = R.Grid(*grid_args)
    t = R.tile_rasters(Path(path), grid, veg_classes)
    tmp = Path(out).with_suffix(".tmp.npz")
    if t is None:
        np.savez_compressed(tmp, empty=True)
        os.replace(tmp, out)
        return 0, 0, time.time() - t0
    dsm = np.where(np.isfinite(t["dsm"]), t["dsm"], np.nan).astype(np.float32)
    np.savez_compressed(tmp, r0=t["r0"], c0=t["c0"], dsm=dsm, nveg=t["nveg"].astype(np.uint16),
                        ntot=t["ntot"].astype(np.uint16), gr0=t["gr0"], gc0=t["gc0"],
                        gmin=np.where(np.isfinite(t["gmin"]), t["gmin"], np.nan).astype(np.float32))
    os.replace(tmp, out)
    chk = np.load(out)  # check: readable, sizes consistent, has returns
    assert chk["dsm"].shape == chk["ntot"].shape and int(chk["ntot"].sum()) > 0
    return int(t["ntot"].sum()), int(np.isfinite(dsm).sum()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--tiles", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--area", choices=["all", "trainval"], default="all")
    a = ap.parse_args()
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # noqa: BLE001
        pass
    if not KEEP_LIST.exists():  # first run: remember every raw file already on disk
        KEEP_LIST.write_text("\n".join(sorted(str(p) for p in T.LIDAR.glob("las*/*") if p.is_file())))
    keep = set(KEEP_LIST.read_text().splitlines())
    grid = city_grid()
    gargs = (grid.x0, grid.y1, grid.nx, grid.ny)
    t = needed(a.year, a.area)
    if a.tiles:
        t = t[t.name.isin(a.tiles)]
    out_dir = OUT / str(a.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = t[[not (out_dir / f"{n}.npz").exists() for n in t.name]]
    if a.limit:
        t = t.head(a.limit)
    log_path = ROOT / f"results/citywide/processing/stream_{a.year}.csv"
    new = not log_path.exists()
    log = open(log_path, "a", newline="")
    w = csv.writer(log)
    if new:
        w.writerow(["tile", "mb", "download_s", "process_s", "points", "cells", "raw_kept", "finished"])
    print(f"{a.year}: {len(t)} tiles to do; grid {grid.nx} x {grid.ny}; workers {a.workers}", flush=True)
    def finish(row_, dt_, fut):
        pts, cells, ps = fut.result()
        kept = str(row_.file) in keep
        if not kept:  # streamed tile: the raster is saved and checked, the raw file goes
            Path(row_.file).unlink(missing_ok=True)
        w.writerow([row_.name, round(row_.bytes / 1e6, 1), round(dt_, 1), round(ps, 1), pts, cells, kept,
                    time.strftime("%F %T")])
        log.flush()
        print(f"  {row_.name}: {row_.bytes / 1e6:.0f} MB, download {dt_:.0f}s, process {ps:.0f}s", flush=True)

    rows = list(t.itertuples())
    with ThreadPoolExecutor(a.workers) as dl, ProcessPoolExecutor(a.workers) as px:
        downloads, processing = [], []
        i = 0
        while i < len(rows) or downloads or processing:
            # keep at most `workers` downloads ahead of processing (bounded disk use)
            while i < len(rows) and len(downloads) + len(processing) < 2 * a.workers:
                while shutil.disk_usage(ROOT).free / 1e9 < MIN_FREE_GB:
                    print(f"disk below {MIN_FREE_GB} GB free: paused", flush=True)
                    time.sleep(300)
                downloads.append(dl.submit(download, rows[i]))
                i += 1
            for f in [f for f in downloads if f.done()]:
                downloads.remove(f)
                row, ok, dt = f.result()
                if not ok:
                    print(f"  download failed: {row.name}", flush=True)
                    continue
                processing.append((row, dt, px.submit(process, str(row.file), str(out_dir / f"{row.name}.npz"),
                                                      gargs, VEG_CLASSES.get(a.year, ()))))
            for q in [q for q in processing if q[2].done()]:
                processing.remove(q)
                finish(*q)
            time.sleep(1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
