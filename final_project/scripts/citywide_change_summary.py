"""Aggregate counts of the citywide LiDAR change labels (train / val ZIPs only; no addresses, no test ZIPs).

    python scripts/citywide_change_summary.py -> results/citywide/change_summary_trainval.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CHG = ROOT / "data/interim/lidar_change_city"

zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
rows = []
for z in zt[~zt.test].zip:
    for f in sorted(CHG.glob(f"{z}_*.parquet")):
        y0, y1 = map(int, f.stem.split("_")[1:])
        d = pd.read_parquet(f)
        d = d[d.in_mask]
        c0, c1 = (0.3 if y0 < 2018 else 0.95), (0.3 if y1 < 2018 else 0.95)
        cov = (d.data_18 > c0) & (d.data_22 > c1) & d.cover18.notna() & d.cover22.notna()
        for t, g in d[cov].groupby("area_type"):
            rows.append(dict(zip=z, period=f"{y0}-{y1}", area_type=t, units=int((d.area_type == t).sum()),
                             covered=len(g), new=int(g.chg_new.sum()), popup=int(g.chg_popup.sum()),
                             demolition=int(g.chg_demo.sum())))
s = pd.DataFrame(rows)
s.to_csv(ROOT / "results/citywide/change_summary_trainval.csv", index=False)
p = s.groupby("period")[["units", "covered", "new", "popup", "demolition"]].sum()
p["new_rate_%"] = (100 * p.new / p.covered).round(2)
p["popup_rate_%"] = (100 * p.popup / p.covered).round(2)
print(p.to_string())
a = s.groupby(["period", "area_type"])[["covered", "new", "popup"]].sum()
print(a.loc[["2018-2022", "2022-2025"]].to_string())
