"""Export georeferenced 2018 / 2022 aerial chips for parcels flagged by the v2 rules (+ all reviewed parcels).

Output: data/interim/chips/{unit}_{year}.png and data/interim/chips/index.json (EPSG:3857 bounds per chip).
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import ortho_chips as O  # noqa: E402
import rear_filter as RF  # noqa: E402

OUT = ROOT / "data/interim/chips"
OUT.mkdir(parents=True, exist_ok=True)
units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet").set_index("unit")
f = RF.apply_v2(C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), C.Rules()),
                pd.read_parquet(ROOT / "data/interim/rear_v2.parquet"))
reviewed = set(pd.read_csv(ROOT / "results/popup_feasibility/review_items.csv").unit)
todo = sorted(set(f[f.chg_any_build_v2 | f.chg_demo].unit) | reviewed)
idx_path = OUT / "index.json"
index = json.loads(idx_path.read_text()) if idx_path.exists() else {}


def one(args):
    u, y = args
    key = f"{u}_{y}"
    if key in index and (OUT / f"{key}.png").exists():
        return key, index[key]
    try:
        img, b = O.chip_georef(units.loc[u, "geometry"], y)
        img.save(OUT / f"{key}.png")
        return key, {"unit": int(u), "year": y, "bounds3857": b, "size": img.size}
    except Exception as e:  # one bad chip must not stop the run; it is retried next time
        print(f"  failed {key}: {e!r}", flush=True)
        return key, None


jobs = [(u, y) for u in todo for y in (2018, 2022)]
with ThreadPoolExecutor(8) as ex:
    for i, (k, v) in enumerate(ex.map(one, jobs), 1):
        if v is not None:
            index[k] = v
        if i % 100 == 0:
            idx_path.write_text(json.dumps(index))  # checkpoint
            print(f"{i}/{len(jobs)}", flush=True)
idx_path.write_text(json.dumps(index))
missing = len(jobs) - sum(f"{u}_{y}" in index for u, y in jobs)
print(f"{len(todo)} parcels, {len(jobs)} chips, {missing} failed")
