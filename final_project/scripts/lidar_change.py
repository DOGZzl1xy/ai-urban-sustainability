"""Stage 1 LiDAR change features for any ZIP and pair of LiDAR years (same rules as the 19146 feasibility test).

    python scripts/lidar_change.py ZIP Y0 Y1   -> data/interim/lidar_change/{ZIP}_{Y0}_{Y1}.parquet

Needs data/interim/units_{ZIP}.parquet (python src/units.py ZIP) and the rasters of both years.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import height_data as H  # noqa: E402
import tiles as T  # noqa: E402
from rasters import Grid  # noqa: E402

ZIP, Y0, Y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def epoch(year):
    z = np.load(H.raster_path(ZIP, year))
    return {k: z[k] for k in ("ndsm", "veg", "ntot")}, Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))


(a, grid), (b, grid2) = epoch(Y0), epoch(Y1)
assert (grid.nx, grid.ny, grid.x0, grid.y1) == (grid2.nx, grid2.ny, grid2.x0, grid2.y1)
units = gpd.read_parquet(ROOT / f"data/interim/units_{ZIP}.parquet")
fp = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                   bbox=tuple(T.study_area((ZIP,)).total_bounds))
f = C.unit_features(units, fp, C.Rules(), epochs=(a, b, grid), sparse=min(Y0, Y1) < 2018)  # 2010/2015 are sparse
out = ROOT / "data/interim/lidar_change"
out.mkdir(parents=True, exist_ok=True)
f.drop(columns="rules").to_parquet(out / f"{ZIP}_{Y0}_{Y1}.parquet")
ok = (f.data_18 > (0.3 if Y0 < 2018 else 0.95)) & (f.data_22 > 0.95)
print(f"{ZIP} {Y0}->{Y1}: {len(f):,} units ({ok.sum():,} fully covered) | new {int(f.chg_new.sum())} | "
      f"pop-up {int(f.chg_popup.sum())} | rear {int(f.chg_rear.sum())} | demolition {int(f.chg_demo.sum())}")
