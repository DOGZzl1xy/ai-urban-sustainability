"""Compare rear-addition rules v1 (initial), v2 (+ contiguity / attachment) and v3 (+ vehicle mask from the
YOLO-OBB detector) on the reviewed sample and on the recall sets.

NOTE: v2/v3 were designed after looking at the reviewed sample, so precision on that sample is optimistic.
Writes results/popup_feasibility/filters_eval.json and data/interim/rear_v3.parquet.
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import rear_filter as RF  # noqa: E402
import tiles as T  # noqa: E402
from match import bootstrap_rate  # noqa: E402

R = ROOT / "results/popup_feasibility"
units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
fp = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                   bbox=tuple(T.study_area().total_bounds))
_, grid = C.load(2018)
vmask, nveh = RF.vehicle_mask(ROOT / "data/interim/chips", grid)
r3 = RF.rear_v2(units, fp, vehicle_mask=vmask).rename(columns={"rear2_m2": "rear3_m2"})
r3.to_parquet(ROOT / "data/interim/rear_v3.parquet")
base = C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), C.Rules())
f = RF.apply_v2(base, pd.read_parquet(ROOT / "data/interim/rear_v2.parquet")).merge(r3, on="unit")
f["chg_rear_v3"] = (f.rear3_m2 >= RF.REAR_M2) & (f.cover18 >= 0.2)
f["v1"] = f.chg_any_build | f.chg_demo
f["v2"] = f.chg_any_build_v2 | f.chg_demo
f["v3"] = f.chg_new | f.chg_popup | f.chg_rear_v3 | f.chg_demo

m = (pd.read_csv(R / "review_items.csv").merge(pd.read_csv(R / "review_labels.csv").fillna(""), on="item")
     .merge(f, on="unit"))
rs = pd.read_parquet(ROOT / "data/interim/recall_set_initial.parquet")[["unit", "kind", "set"]].merge(f, on="unit")
rs = rs[rs.set == "primary"]
out = {"vehicles_detected": nveh, "flag_counts": {}, "reviewed": {}, "recall": {}}
for v in ("v1", "v2", "v3"):
    out["flag_counts"][v] = int(f[v].sum())
    p = m[(m.role == "precision") & m[v] & m.change.isin(["real", "none"])]
    x = (p.change == "real").to_numpy(float)
    u = m[m.role == "unpermitted"]
    out["reviewed"][v] = {
        "precision_sample_still_flagged": int(m[(m.role == "precision")][v].sum()),
        "precision_on_judged": float(x.mean()) if len(x) else None, "precision_ci": bootstrap_rate(x),
        "false_pos_remaining_all_reviewed": int((m[v] & (m.change == "none")).sum()),
        "true_changes_kept_all_reviewed": int((m[v] & (m.change == "real")).sum()),
        "true_changes_total_reviewed": int((m.change == "real").sum()),
        "confirmed_unpermitted_kept": int((u[v] & (u.change == "real") & (u.permit == "no_permit")).sum()),
    }
    out["recall"][v] = {k: float(rs[rs.kind == k][v].mean()) for k in ("new", "upward", "rear")}
(R / "filters_eval.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
fp_left = m[m.v3 & (m.change == "none")][["item", "change_note", "chg_new", "chg_popup", "chg_rear_v3"]]
print("\nfalse positives remaining under v3:\n", fp_left.to_string(index=False))
