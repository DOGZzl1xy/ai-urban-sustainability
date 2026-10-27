"""Build the 0.5 m DSM / DTM / nDSM / vegetation rasters for one epoch over the study area.

Usage: python build_rasters.py YEAR [WORKERS] [ZIP]
  -> data/interim/rasters_{YEAR}.npz for 19146 (feasibility test), data/interim/rasters_{ZIP}_{YEAR}.npz otherwise
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rasters as R  # noqa: E402
import tiles as T  # noqa: E402

YEAR = int(sys.argv[1])
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
ZIP = sys.argv[3] if len(sys.argv) > 3 else "19146"
GRID = R.Grid.around(T.study_area((ZIP,)).total_bounds)
OUT = ROOT / (f"data/interim/rasters_{YEAR}.npz" if ZIP == "19146" else f"data/interim/rasters_{ZIP}_{YEAR}.npz")


def work(path):
    t = time.time()
    return path, R.tile_rasters(Path(path), GRID), time.time() - t


if __name__ == "__main__":
    tl = T.study_tiles(YEAR, (ZIP,))
    paths = [str(f) for f, b in zip(tl.file, tl.bytes) if Path(f).exists() and Path(f).stat().st_size == b]
    print(f"{ZIP} {YEAR}: {len(paths)}/{len(tl)} tiles on disk; grid {GRID.nx} x {GRID.ny} cells", flush=True)
    m = R.Mosaic(GRID)
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, f in enumerate(as_completed([ex.submit(work, p) for p in paths]), 1):
            p, t, dt = f.result()
            m.add(t)
            print(f"  {i}/{len(paths)} {Path(p).name} {dt:.0f}s", flush=True)
    dsm, dtm, ndsm, veg, have = m.finish()
    np.savez_compressed(OUT, dsm=dsm, dtm=dtm, ndsm=ndsm, veg=veg,
                        gmin=m.gmin, ntot=m.ntot, x0=GRID.x0, y1=GRID.y1, nx=GRID.nx, ny=GRID.ny)
    cov = np.isfinite(dsm).mean()
    print(f"done: DSM coverage {cov:.1%}, median pts/cell {np.median(m.ntot[m.ntot > 0]):.0f}")
