"""Classify the model's false pop-ups on the development test (19146 + 19147, 2018 -> 2022) by what LiDAR shows.

    python scripts/fp_classify.py TAG [--review N]

False pop-up = the model flags a pop-up, the Stage 1 LiDAR rules do not. Categories (first match wins):
  (a) new construction    LiDAR flags the unit as new construction
  (b) near the threshold  median rise 1.5-2.5 m on the 2018 building, or 5-30% of the 2018 building rose >= 2.5 m
  (c) rear rise           new building area outside the 2017 footprint >= 10 m2 (rear addition)
  (d) no change           none of the above: |median dh| < 0.5 m and < 5% of the building rose
  (e) other               anything else (e.g. demolition, vegetation, partial data)
Writes results/bitemporal/fp_classes_{TAG}.csv (counts) and, for category (d) only, a review page with at most N
(default 30) random units: results/bitemporal/review_fp_d_{TAG}.html (local only: it links street addresses).
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--review", type=int, default=30)
ap.add_argument("--seed", type=int, default=20261023)
a = ap.parse_args()
RES = ROOT / "results/bitemporal"

d = pd.read_parquet(ROOT / f"data/interim/unitcls_preds/{a.tag}_dev_2018_2022.parquet")
lc = pd.concat([pd.read_parquet(ROOT / f"data/interim/lidar_change/{z}_2018_2022.parquet").assign(zip=z)
                for z in M.TEST_ZIPS])
d = d.merge(lc, on=["zip", "unit"], how="left")
fp = d[(d.logit_popup >= d.thr_popup) & ~d.popup.astype(bool)].copy()
fp["frac_up"] = fp.frac_up.fillna(0)
fp["dh_p50"] = fp.dh_p50.fillna(0)
cat = np.select(
    [fp.chg_new.fillna(False),
     fp.dh_p50.between(1.5, 2.5) | fp.frac_up.between(0.05, 0.3, inclusive="left"),
     fp.new_outside_fp_m2.fillna(0) >= 10,
     (fp.dh_p50.abs() < 0.5) & (fp.frac_up < 0.05)],
    ["a_new_construction", "b_near_threshold", "c_rear_rise", "d_no_change"], default="e_other")
fp["category"] = cat
counts = fp.groupby(["category", "zip"]).size().unstack(fill_value=0)
counts["total"] = counts.sum(1)
counts.loc["all"] = counts.sum()
counts.to_csv(RES / f"fp_classes_{a.tag}.csv")
print(f"{a.tag}: {len(fp)} false pop-ups (model flags pop-up, LiDAR does not)")
print(counts.to_string())

dd = fp[fp.category == "d_no_change"]
if len(dd) and a.review:
    import review_page as RP

    s = dd.sample(min(a.review, len(dd)), random_state=a.seed)
    units = pd.concat([gpd.read_parquet(ROOT / f"data/interim/units_{z}.parquet")[["unit", "address", "geometry"]]
                       .assign(zip=z) for z in s.zip.unique()])
    s = s.merge(units, on=["zip", "unit"])
    s["item"] = [f"D{i + 1:02d}" for i in range(len(s))]
    s["role"] = "precision"
    for c in ("chg_new", "chg_popup", "chg_rear", "chg_demo"):
        s[c] = s[c].fillna(False).astype(bool)
    s[["item", "zip", "unit"]].to_csv(RES / f"review_fp_d_{a.tag}_items.csv", index=False)
    html = RP.build(s.reset_index(drop=True), f"False pop-ups with no LiDAR change ({a.tag})",
                    f"fp-d-{a.tag}", "The model flagged these units as pop-ups (2018 to 2022), but LiDAR shows no "
                    "height change. What does the image show (roof equipment, shadow, construction, nothing)?",
                    export_name=f"review_fp_d_{a.tag}_labels.csv")
    (RES / f"review_fp_d_{a.tag}.html").write_text(html, encoding="utf-8")
    print(f"review page: {len(s)} units -> review_fp_d_{a.tag}.html")
