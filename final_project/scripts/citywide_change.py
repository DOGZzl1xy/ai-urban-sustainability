"""Citywide LiDAR change per unit, ZIP by ZIP, from the streamed citywide tiles (same v3 rules for new construction
and pop-ups as the 19146 feasibility test; labels use only those two classes).

    python scripts/citywide_change.py [--zips Z ...] [--role trainval|test] [--workers 3] [--mem-gb 12]

For each ZIP (window = ZIP + 300 ft, snapped to the citywide 0.5 m grid):
  rasters  data/interim/citywide_rasters/{ZIP}_{YEAR}.npz   (same keys as rasters_{ZIP}_{YEAR}.npz)
  units    data/interim/citywide_units/{ZIP}.parquet        (src/units.py build + in_mask, area_type)
  change   data/interim/lidar_change_city/{ZIP}_{Y0}_{Y1}.parquet
Periods: 2018-2022 and 2022-2025 everywhere; the M4 training periods with 2010 / 2015 in train / val ZIPs only
(2010 / 2015 were streamed for the train / val area only).
Test ZIPs (role test) are not run by default: their labels are only built for the round-2 final test, and nothing
about them is printed or summarised before M4 is frozen.
The per-ZIP files used in Stage 3 (rasters_*, units_*, lidar_change/) are left untouched.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import change as C  # noqa: E402
import rasters as R  # noqa: E402
import tiles as T  # noqa: E402
import units as U  # noqa: E402
from stream_lidar import city_grid  # noqa: E402

TILES = ROOT / "data/interim/citywide_tiles"
RAS = ROOT / "data/interim/citywide_rasters"
UNITS = ROOT / "data/interim/citywide_units"
CHG = ROOT / "data/interim/lidar_change_city"
FULL_PERIODS = ((2018, 2022), (2022, 2025))
TRAIN_PERIODS = ((2010, 2015), (2010, 2018), (2010, 2022), (2015, 2018), (2015, 2022))
TILE_INDEX = {2010: T.tiles_2010, 2015: T.tiles_2015, 2018: T.tiles_2018, 2022: T.tiles_2022, 2025: T.tiles_2025}


def window(zip_code, city):
    b = T.study_area((zip_code,), buffer_ft=300).total_bounds
    c0 = int(np.floor((b[0] - city.x0) / R.CELL))
    r0 = int(np.floor((city.y1 - b[3]) / R.CELL))
    c1 = int(np.ceil((b[2] - city.x0) / R.CELL))
    r1 = int(np.ceil((city.y1 - b[1]) / R.CELL))
    c0, r0 = c0 - c0 % R.DTM_FACTOR, r0 - r0 % R.DTM_FACTOR   # keep the 1.5 m ground grid aligned with the city's
    return r0, c0, R.Grid(city.x0 + c0 * R.CELL, city.y1 - r0 * R.CELL, c1 - c0, r1 - r0)


def fill_dtm(gm, chunk=2000, halo=200):
    """Mosaic.finish's ground fill (linear, then nearest for holes) in blocks of chunk x chunk 1.5 m cells (3 km)
    with a halo of 200 cells (300 m): one Delaunay over a whole large ZIP needs > 20 GB. Windows up to 3 km x 3 km
    are filled in one block, exactly as Mosaic.finish does."""
    from scipy.interpolate import griddata
    out = np.empty(gm.shape, np.float64)
    H, W = gm.shape
    for i in range(0, H, chunk):
        for j in range(0, W, chunk):
            a0, a1 = max(i - halo, 0), min(i + chunk + halo, H)
            b0, b1 = max(j - halo, 0), min(j + chunk + halo, W)
            if H <= chunk and W <= chunk:
                a0, a1, b0, b1 = 0, H, 0, W
            sub = gm[a0:a1, b0:b1]
            have = np.isfinite(sub)
            if have.sum() < 3:   # no ground in the block and its halo: leave it to the global nearest fill below
                out[i:i + chunk, j:j + chunk] = np.nan
                continue
            rr, cc = np.nonzero(have)
            gy, gx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
            f = griddata((rr, cc), sub[have], (gy, gx), method="linear")
            hole = ~np.isfinite(f)
            if hole.any():
                f[hole] = griddata((rr, cc), sub[have], (gy[hole], gx[hole]), method="nearest")
            out[i:i + chunk, j:j + chunk] = f[i - a0:i - a0 + min(chunk, H - i), j - b0:j - b0 + min(chunk, W - j)]
    if np.isnan(out).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(out), return_distances=False, return_indices=True)
        out = out[tuple(idx)]
    return out


def finish(m):
    """Mosaic.finish with the chunked ground fill."""
    from scipy.ndimage import zoom
    filled = fill_dtm(m.gmin)
    dtm = zoom(filled, R.DTM_FACTOR, order=1)[: m.g.ny, : m.g.nx].astype(np.float32)
    dsm = np.where(np.isfinite(m.dsm), m.dsm, np.nan)
    ndsm = ((dsm - dtm) * R.FT).astype(np.float32)
    veg = np.where(m.ntot > 0, m.nveg / np.maximum(m.ntot, 1), np.nan).astype(np.float32)
    return dsm, dtm, ndsm, veg


def build_raster(zip_code, year):
    out = RAS / f"{zip_code}_{year}.npz"
    if out.exists():
        return f"{zip_code} {year}: exists"
    t0 = time.time()
    city = city_grid()
    r0, c0, g = window(zip_code, city)
    ti = TILE_INDEX[year]()
    area = T.study_area((zip_code,), buffer_ft=300).union_all()
    names = [n for n in ti[ti.intersects(area)].name if (TILES / str(year) / f"{n}.npz").exists()]
    m = R.Mosaic(g)
    for n in names:
        z = np.load(TILES / str(year) / f"{n}.npz")
        if "empty" in z.files:
            continue
        t = {k: z[k] for k in z.files}
        # shift from citywide to window indices and clip to the window
        rr, cc = int(t["r0"]) - r0, int(t["c0"]) - c0
        h, w = t["dsm"].shape
        a0, a1 = max(rr, 0), min(rr + h, g.ny)
        b0, b1 = max(cc, 0), min(cc + w, g.nx)
        if a0 >= a1 or b0 >= b1:
            continue
        sl = (slice(a0 - rr, a1 - rr), slice(b0 - cc, b1 - cc))
        dsm = np.where(np.isnan(t["dsm"][sl]), -np.inf, t["dsm"][sl])
        gr, gc = int(t["gr0"]) - r0 // R.DTM_FACTOR, int(t["gc0"]) - c0 // R.DTM_FACTOR
        gh, gw = t["gmin"].shape
        GH, GW = m.gmin.shape
        ga0, ga1, gb0, gb1 = max(gr, 0), min(gr + gh, GH), max(gc, 0), min(gc + gw, GW)
        gm = np.where(np.isnan(t["gmin"]), np.inf, t["gmin"])
        m.add(dict(r0=a0, c0=b0, dsm=dsm, nveg=t["nveg"][sl].astype(np.int32), ntot=t["ntot"][sl].astype(np.int32),
                   gr0=ga0, gc0=gb0, gmin=gm[ga0 - gr:ga1 - gr, gb0 - gc:gb1 - gc]))
    dsm, dtm, ndsm, veg = finish(m)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, dsm=dsm, dtm=dtm, ndsm=ndsm, veg=veg, gmin=m.gmin, ntot=m.ntot,
                        x0=g.x0, y1=g.y1, nx=g.nx, ny=g.ny, r0_city=r0, c0_city=c0)
    tmp.replace(out)
    return f"{zip_code} {year}: {len(names)} tiles, {g.nx} x {g.ny}, coverage {np.isfinite(dsm).mean():.0%}, " \
           f"{time.time() - t0:.0f}s"


def build_units(zip_code):
    out = UNITS / f"{zip_code}.parquet"
    if out.exists():
        return gpd.read_parquet(out)
    u = U.build(T.study_area((zip_code,), buffer_ft=0))
    d = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="districts")[["area_type", "geometry"]]
    pt = gpd.GeoDataFrame(u[["unit"]], geometry=u.representative_point(), crs=2272)
    j = gpd.sjoin(pt, d, how="left", predicate="within").drop_duplicates("unit").set_index("unit")
    u["area_type"] = u.unit.map(j.area_type)
    u["in_mask"] = u.area_type.notna()
    u.to_parquet(out)
    return u


def epoch(zip_code, year):
    z = np.load(RAS / f"{zip_code}_{year}.npz")
    return {k: z[k] for k in ("ndsm", "veg", "ntot")}, R.Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]),
                                                              int(z["ny"]))


def change(zip_code, y0, y1):
    out = CHG / f"{zip_code}_{y0}_{y1}.parquet"
    if out.exists():
        return f"{zip_code} {y0}-{y1}: exists"
    units = build_units(zip_code)
    (a, g), (b, g2) = epoch(zip_code, y0), epoch(zip_code, y1)
    assert (g.nx, g.ny, g.x0, g.y1) == (g2.nx, g2.ny, g2.x0, g2.y1)
    fp = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                       bbox=tuple(T.study_area((zip_code,)).total_bounds))
    f = C.unit_features(units, fp, C.Rules(), epochs=(a, b, g), sparse=min(y0, y1) < 2018)
    f = f.drop(columns="rules").merge(units[["unit", "in_mask", "area_type"]], on="unit")
    f.to_parquet(out)
    return f"{zip_code} {y0}-{y1}: done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", nargs="*")
    ap.add_argument("--role", choices=["trainval", "test"], default="trainval")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mem-gb", type=float, default=12)
    a = ap.parse_args()
    for d in (RAS, UNITS, CHG):
        d.mkdir(parents=True, exist_ok=True)
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = a.zips or list(zt[zt.test == (a.role == "test")].zip)
    periods = FULL_PERIODS + (TRAIN_PERIODS if a.role == "trainval" else ())
    years = sorted({y for p in periods for y in p})
    # memory: a raster job peaks at ~50 bytes per window cell (19128, 203 M cells: ~10 GB), a change job at ~60.
    # Jobs start largest first while the estimated memory in flight stays under --mem-gb.
    city = city_grid()
    cells = {z: window(z, city)[2].nx * window(z, city)[2].ny for z in zips}
    quiet = a.role == "test"

    def run(fn, jobs, bytes_per_cell):
        jobs = sorted(jobs, key=lambda j: -cells[j[0]])
        running = {}
        with ProcessPoolExecutor(a.workers) as ex:
            while jobs or running:
                used = sum(running.values())
                for j in list(jobs):
                    need = cells[j[0]] * bytes_per_cell / 1e9
                    if len(running) < a.workers and (not running or used + need <= a.mem_gb):
                        running[ex.submit(fn, *j)] = need
                        used += need
                        jobs.remove(j)
                for f in [f for f in running if f.done()]:
                    del running[f]
                    msg = f.result()
                    if not quiet:
                        print(" ", msg, flush=True)
                time.sleep(1)

    run(build_raster, [(z, y) for z in zips for y in years], 50)
    for z in zips:   # units once per ZIP (sequential: cheap, avoids two processes writing one file)
        build_units(z)
    run(change, [(z, y0, y1) for z in zips for y0, y1 in periods], 60)
    print("CHANGE DONE", flush=True)


if __name__ == "__main__":
    main()
