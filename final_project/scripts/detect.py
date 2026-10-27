"""Step 1: per-unit change features + recall of permitted work under a rule set.

Usage: python detect.py [rules.json]   (no argument = the initial thresholds from FEASIBILITY_TEST.md)
Writes data/interim/unit_features.parquet and results/popup_feasibility/recall_<tag>.csv
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import match as M  # noqa: E402
import permits as P  # noqa: E402
import tiles as T  # noqa: E402

OUT = ROOT / "results/popup_feasibility"
DATES = json.loads((OUT / "lidar_dates.json").read_text())
D18, D22 = pd.Timestamp(DATES["2018"]["first"]), pd.Timestamp(DATES["2022"]["first"])

rules_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
rules = C.Rules(**json.loads(rules_path.read_text())) if rules_path else C.Rules()
tag = rules_path.stem if rules_path else "initial"

units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
feat_path = ROOT / "data/interim/unit_features.parquet"
if feat_path.exists() and rules.bldg_h == 2.5 and rules.up_h == 2.5 and rules.veg_major == 0.5:
    f = C.apply_rules(pd.read_parquet(feat_path), rules)
else:
    fp = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                       bbox=tuple(T.study_area().total_bounds))
    f = C.unit_features(units, fp, rules)
    if tag == "initial":
        f.to_parquet(feat_path)

p = M.match_permits(P.classify(P.download(T.study_area())), units)
p = p[p.unit.notna()].copy()
p["unit"] = p.unit.astype(int)
rs = M.recall_sets(p, D18, D22).merge(f, on="unit", how="left")
rs["detected_any"] = rs[["chg_new", "chg_popup", "chg_rear", "chg_demo"]].any(axis=1)
rs["detected_type"] = np.select([rs.kind == "new", rs.kind == "upward", rs.kind == "rear"],
                                [rs.chg_new | rs.chg_popup, rs.chg_popup, rs.chg_rear])
rows = []
for (kind, s), g in rs.groupby(["kind", "set"]):
    for col in ("detected_any", "detected_type"):
        x = g[col].fillna(False).to_numpy(dtype=float)
        lo, hi = M.bootstrap_rate(x)
        rows.append({"kind": kind, "set": s, "measure": col, "n": len(x), "recall": x.mean(), "lo": lo, "hi": hi})
rec = pd.DataFrame(rows)
rec.to_csv(OUT / f"recall_{tag}.csv", index=False)
rs.to_parquet(ROOT / f"data/interim/recall_set_{tag}.parquet")
print(f"rules [{tag}]: {rules}")
print(f"LiDAR dates: 2018 from {D18.date()}, 2022 from {D22.date()}")
print(rec.round(3).to_string(index=False))
print("\nflag counts:", {c: int(f[c].sum()) for c in ["chg_new", "chg_demo", "chg_popup", "chg_rear", "chg_any_build"]},
      "of", len(f), "units")
# diagnostics for the single allowed revision: feature distributions of missed vs detected primary cases
prim = rs[rs.set == "primary"].copy()  # issued and completed between epochs
prim["detected_type"] = prim.detected_type.fillna(False).astype(bool)
for kind, cols in [("new", ["cover18", "cover22"]), ("upward", ["frac_up", "dh_p50", "cover18"]),
                   ("rear", ["new_outside_fp_m2"])]:
    g = prim[prim.kind == kind]
    print(f"\n{kind}: quantiles of {cols} — missed (n={(~g.detected_type).sum()}) vs detected (n={g.detected_type.sum()})")
    for det, gg in g.groupby("detected_type"):
        print("  detected" if det else "  missed  ", gg[cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).to_dict())
