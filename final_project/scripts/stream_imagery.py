"""Cache the city's aerial imagery tiles (zoom 19, ~0.23 m) over the study area for the citywide model.

    python scripts/stream_imagery.py [--years 2018 2019 ...] [--rate 10] [--threads 4] [--count-only]

Years 2018, 2019, 2020, 2022, 2023, 2024, 2025 cover the whole study area (+150 m); 2010 and 2015 cover the train /
validation area only (training pairs). Tiles go to the existing cache data/raw/ortho/tiles/{year}/19_{x}_{y}.jpg
(JPEG as served, so already compressed); cached tiles are skipped, so the run is resumable. Requests are throttled
to --rate per second across all threads (the service is public, allows tile export, and the item asks bulk users of
full-resolution TIFFs to use PASDA; zoom 19 is far below that). Log: results/citywide/processing/imagery_{year}.csv.
"""

import argparse
import csv
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import STRtree, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import ortho_chips as O  # noqa: E402

Z = 19
FULL = (2018, 2019, 2020, 2022, 2023, 2024, 2025)
TRAINVAL_ONLY = (2010, 2015)


def tile_list(area_3857):
    r = O._res(Z) * 256
    x0, y0, x1, y1 = area_3857.bounds
    tx0, tx1 = int((x0 + O.ORIGIN) // r), int((x1 + O.ORIGIN) // r)
    ty0, ty1 = int((O.ORIGIN - y1) // r), int((O.ORIGIN - y0) // r)
    cand = [(tx, ty) for tx in range(tx0, tx1 + 1) for ty in range(ty0, ty1 + 1)]
    boxes = [box(tx * r - O.ORIGIN, O.ORIGIN - (ty + 1) * r, (tx + 1) * r - O.ORIGIN, O.ORIGIN - ty * r)
             for tx, ty in cand]
    parts = list(area_3857.geoms) if hasattr(area_3857, "geoms") else [area_3857]
    hit = STRtree(boxes).query(parts, predicate="intersects")[1]
    return [cand[i] for i in np.unique(hit)]


class Throttle:
    def __init__(self, rate):
        self.gap, self.lock, self.next = 1.0 / rate, threading.Lock(), time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            t = max(now, self.next)
            self.next = t + self.gap
        time.sleep(max(0.0, t - now))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=list(FULL + TRAINVAL_ONLY))
    ap.add_argument("--rate", type=float, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--count-only", action="store_true")
    a = ap.parse_args()
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask")
    full = mask.to_crs(3857).buffer(150).union_all()
    bl = gpd.read_file(ROOT / "data/interim/splits_citywide.gpkg", layer="blocks")
    tv = gpd.GeoSeries([bl[bl.split != "test"].union_all().intersection(mask.union_all())], crs=2272)
    trainval = tv.to_crs(3857).buffer(150).union_all()
    lists = {"full": tile_list(full), "trainval": tile_list(trainval)}
    throttle = Throttle(a.rate)
    for y in a.years:
        tl = lists["trainval" if y in TRAINVAL_ONLY else "full"]
        todo = [(tx, ty) for tx, ty in tl if not (O.CACHE / str(y) / f"{Z}_{tx}_{ty}.jpg").exists()]
        print(f"{y}: {len(tl)} tiles, {len(todo)} to download (~{len(todo) / a.rate / 3600:.1f} h at {a.rate}/s)",
              flush=True)
        if a.count_only or not todo:
            continue
        log = open(ROOT / f"results/citywide/processing/imagery_{y}.csv", "a", newline="")
        w = csv.writer(log)
        ok = fail = 0
        t0 = time.time()

        def one(t):
            throttle.wait()
            return O._fetch(y, Z, t[0], t[1], O.CACHE / str(y) / f"{Z}_{t[0]}_{t[1]}.jpg")

        with ThreadPoolExecutor(a.threads) as ex:
            for k, r in enumerate(ex.map(one, todo), 1):
                ok, fail = ok + bool(r), fail + (not r)
                if k % 2000 == 0 or k == len(todo):
                    w.writerow([y, k, len(todo), ok, fail, round(time.time() - t0)])
                    log.flush()
                    print(f"  {y}: {k}/{len(todo)} ({fail} missing/failed) {time.time() - t0:.0f}s", flush=True)
    print("IMAGERY DONE", flush=True)


if __name__ == "__main__":
    main()
