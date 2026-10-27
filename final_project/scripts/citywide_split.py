"""Second-round sealed test area and citywide train / validation split. Run ONCE; the result is committed and frozen.

    python scripts/citywide_split.py

1. ZIP codes are typed by the area type that covers most of their study-area land (results/citywide/
   zip_area_types.csv): rowhouse, detached or mixed.
2. Eligible for the test draw: >= 0.5 km2 of study area, not one of the 7 Stage 3 training ZIPs, not 19146 / 19147.
3. Per type, round(22% x eligible) ZIPs (at least 1) are drawn with numpy default_rng(SEED) from the sorted list.
4. 1 km blocks (same grid as bitemporal.make_splits_bt: floor(x / 1 km), floor(y / 1 km) in EPSG:2272) inside the
   study area: blocks whose study-area centre lies in a test ZIP -> 'test'; of the rest, 15% drawn at random (same
   seed) -> 'val', the others 'train'.
5. Patch rules (applied when patches are cut): a patch belongs to the block of its centre; a patch touching a test
   ZIP buffered by 100 m never enters training or validation; a train patch within 50 m of a val block is dropped.
   Test evaluation uses only units inside the test ZIPs.
Writes data/interim/splits_citywide.gpkg (layers zips, blocks) and results/citywide/splits_citywide.csv.
"""

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261023
TEST_SHARE, VAL_SHARE, MIN_KM2 = 0.22, 0.15, 0.5
TRAINED = {"19132", "19121", "19143", "19125", "19130", "19123", "19122"}
ROUND1_TEST = {"19146", "19147"}
FT = 0.3048
BLK = 1000 / FT

rng = np.random.default_rng(SEED)
t = pd.read_csv(ROOT / "results/citywide/zip_area_types.csv", dtype={"zip": str}).set_index("zip")
elig = t[(t.total >= MIN_KM2) & ~t.index.isin(TRAINED | ROUND1_TEST)]
test = []
for typ in ("rowhouse", "detached", "mixed"):
    pool = sorted(elig[elig.type == typ].index)
    k = max(1, int(math.floor(TEST_SHARE * len(pool) + 0.5)))
    test += list(rng.choice(pool, k, replace=False))
zips = t.assign(eligible=t.index.isin(elig.index), test=t.index.isin(test),
                role=np.where(t.index.isin(test), "test",
                              np.where(t.index.isin(TRAINED), "train_val (Stage 3 training ZIP)",
                                       np.where(t.index.isin(ROUND1_TEST), "train_val (round-1 test ZIP)",
                                                "train_val"))))

zg = gpd.read_file(ROOT / "data/raw/boundaries/zip_codes.geojson").to_crs(2272)
zg["zip"] = zg.code.astype(str)
mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
test_buf = zg[zg.zip.isin(test)].union_all().buffer(100 / FT)
x0, y0, x1, y1 = mask.bounds
blocks = []
for bx in range(int(x0 // BLK), int(x1 // BLK) + 1):
    for by in range(int(y0 // BLK), int(y1 // BLK) + 1):
        b = box(bx * BLK, by * BLK, (bx + 1) * BLK, (by + 1) * BLK)
        a = b.intersection(mask).area
        if a > 0:
            blocks.append({"block": f"{bx}_{by}", "study_km2": a * FT ** 2 / 1e6, "geometry": b})
blocks = gpd.GeoDataFrame(blocks, crs=2272)
test_zone = zg[zg.zip.isin(test)].union_all()
cent = [b.intersection(mask).representative_point() for b in blocks.geometry]
blocks["split"] = np.where([c.within(test_zone) for c in cent], "test", "train")
blocks["touches_test_buffer"] = blocks.intersects(test_buf)
cand = blocks.index[blocks.split == "train"]
val = rng.choice(cand, int(round(VAL_SHARE * len(cand))), replace=False)
blocks.loc[val, "split"] = "val"
vb = blocks[blocks.split == "val"].union_all().buffer(50 / FT)
blocks["next_to_val"] = (blocks.split == "train") & blocks.geometry.buffer(-1).intersects(vb)
out = ROOT / "data/interim/splits_citywide.gpkg"
zg.merge(zips.reset_index()[["zip", "type", "total", "eligible", "test", "role"]], on="zip").to_file(
    out, layer="zips", driver="GPKG")
blocks.to_file(out, layer="blocks", driver="GPKG")
blocks.drop(columns="geometry").round(3).to_csv(ROOT / "results/citywide/splits_citywide.csv", index=False)
zips.round(3).to_csv(ROOT / "results/citywide/splits_citywide_zips.csv")
print("test ZIPs:", sorted(test))
print(zips[zips.test][["type", "total"]].round(2).to_string())
print(blocks.groupby("split").agg(blocks=("block", "size"), study_km2=("study_km2", "sum")).round(1).to_string())
