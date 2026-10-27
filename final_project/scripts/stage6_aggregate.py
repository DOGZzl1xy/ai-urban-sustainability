"""Stage 6 aggregation (results/citywide/stage6_plan.md), local: LiDAR anchor periods + model intermediate years.

    POPUP_DATA=citywide python scripts/stage6_aggregate.py

Inputs: data/interim/stage6_scores/{ZIP}_{A}_{T}.parquet (cloud, scripts/stage6.py infer), LiDAR labels
data/interim/lidar_change_city/{ZIP}_{y0}_{y1}.parquet, units data/interim/citywide_units (centroid, mask, area type),
results/citywide/highrise_all.csv, M4 thresholds (unitcls_m4s_s0_val*.json) and count calibration (round2_test.json,
from M4's validation units).
Units: inside the study-area mask. LiDAR counts use units covered by LiDAR in both years (the labelling rule); model
estimates use the units the model scored; estimate = flags x validation precision / recall.
Outputs (results/citywide/stage6/): cells_{100m,250m,bg}.csv (one row per cell: units, coverage, LiDAR counts per
period, model estimates per pair, cumulative series since 2018, high-rise flag), totals.csv (study-area totals by
year, main cells and high-rise cells), consistency.csv (model vs LiDAR ordering per cell).
"""

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
from rasters import FT  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
OUT = ROOT / "results/citywide/stage6"
SCORES = ROOT / "data/interim/stage6_scores"
PAIRS = ((2018, 2019), (2018, 2020), (2022, 2023), (2022, 2024))
PERIODS = ((2018, 2022), (2022, 2025))
SCALES = {"c100": "100m", "c250": "250m", "bg": "bg"}


def units_geo(zips):
    u = pd.concat([gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")[
        ["unit", "geometry", "in_mask", "area_type"]].assign(zip=z) for z in zips])
    u = gpd.GeoDataFrame(u, geometry="geometry", crs=2272)
    u = u[u.in_mask].copy()
    c = u.geometry.centroid
    for m in (100, 250):
        s = m / FT
        u[f"c{m}"] = [f"{int(x // s)}_{int(y // s)}" for x, y in zip(c.x, c.y)]
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2019_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    bg = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2019_42_bg.zip'}!{name}")
    bg = bg[bg.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]]
    j = gpd.sjoin(gpd.GeoDataFrame(u[["zip", "unit"]], geometry=c, crs=2272), bg, how="left", predicate="within")
    u["bg"] = j.GEOID.to_numpy()
    return pd.DataFrame(u.drop(columns="geometry"))


def lidar(zips, y0, y1):
    rows = []
    for z in zips:
        f = pd.read_parquet(ROOT / f"data/interim/lidar_change_city/{z}_{y0}_{y1}.parquet")
        c0, c1 = (0.3 if y0 < 2018 else 0.95), (0.3 if y1 < 2018 else 0.95)
        cov = (f.data_18 > c0) & (f.data_22 > c1) & f.cover18.notna() & f.cover22.notna()
        rows.append(pd.DataFrame({"zip": z, "unit": f.unit, "covered": cov.to_numpy(),
                                  "new": f.chg_new.astype(bool), "popup": f.chg_popup.astype(bool),
                                  "demo": f.chg_demo.astype(bool)}))
    d = pd.concat(rows)
    d["grew"] = d.new | d.popup
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = sorted(zt.zip)
    idx = pd.read_csv(ROOT / "data/interim/stage6_patches.csv", dtype={"zip": str})
    has = set(zip(idx.zip, idx.anchor))   # a ZIP-anchor without any patch (19102: too small) has no scores
    miss = [f"{z}_{a}_{t}" for z in zips for a, t in PAIRS
            if (z, a) in has and not (SCORES / f"{z}_{a}_{t}.parquet").exists()]
    assert not miss, f"stage-6 scores missing: {miss[:5]} ({len(miss)})"
    thr = {"popup": json.load(open(ROOT / "results/citywide/m4/unitcls_m4s_s0_val.json"))["thresholds"]["popup"],
           "grew": json.load(open(ROOT / "results/citywide/m4/unitcls_m4s_s0_val_grew.json"))["thresholds"]["grew"]}
    cal = json.load(open(ROOT / "results/citywide/m4/round2_test.json"))["models"]["m4s_s0"]["calibration_from_validation"]
    fac = {k: cal[k]["factor"] for k in ("new", "grew")}
    u = units_geo(zips)
    base = u.copy()
    # LiDAR anchor periods
    for y0, y1 in PERIODS:
        p = f"{y0}_{y1}"
        d = lidar(zips, y0, y1)
        d = d[d.covered].rename(columns={k: f"L{p}_{k}" for k in ("new", "popup", "grew", "demo")})
        base = base.merge(d.drop(columns="covered").assign(**{f"cov{p}": True}), on=["zip", "unit"], how="left")
        base[f"cov{p}"] = base[f"cov{p}"].fillna(False).astype(bool)
    # model intermediate years
    for a, t in PAIRS:
        p = f"{a}_{t}"
        s = pd.concat([pd.read_parquet(SCORES / f"{z}_{a}_{t}.parquet") for z in zips if (z, a) in has])
        s[f"M{p}_grew"] = s.logit_grew >= thr["grew"]
        s[f"M{p}_new"] = s[f"M{p}_grew"] & (s.logit_popup < thr["popup"])
        s[f"sc{p}"] = True
        base = base.merge(s[["zip", "unit", f"sc{p}", f"M{p}_grew", f"M{p}_new"]], on=["zip", "unit"], how="left")
        base[f"sc{p}"] = base[f"sc{p}"].fillna(False).astype(bool)
    hr = pd.read_csv(ROOT / "results/citywide/highrise_all.csv", dtype={"key": str})
    flagged = {k: set(hr[(hr.unit == k) & hr.flagged].key) for k in SCALES}
    for k in SCALES:
        base[f"hr_{k}"] = base[k].isin(flagged[k])
    base.to_parquet(ROOT / "data/interim/stage6_units.parquet")   # per unit (unit ids, cells, flags; no addresses)
    totals, cons = [], []
    for key, name in SCALES.items():
        g = base.dropna(subset=[key]).groupby(key)
        c = pd.DataFrame({"units": g.size()})
        for y0, y1 in PERIODS:
            p = f"{y0}_{y1}"
            c[f"lidar_cov_{p}"] = g[f"cov{p}"].mean()
            for k in ("new", "popup", "grew", "demo"):
                c[f"lidar_{k}_{p}"] = g[f"L{p}_{k}"].apply(lambda x: x.fillna(False).astype(bool).sum())
        for a, t in PAIRS:
            p = f"{a}_{t}"
            c[f"model_cov_{p}"] = g[f"sc{p}"].mean()
            for k in ("new", "grew"):
                flags = g[f"M{p}_{k}"].apply(lambda x: x.fillna(False).astype(bool).sum())
                c[f"model_flags_{k}_{p}"] = flags
                c[f"model_est_{k}_{p}"] = flags * fac[k]
        # cumulative change since each anchor, on ONE unit set per anchor (LiDAR-covered over the anchor period AND
        # scored by the model in both target years): series A = 2018 anchor (2019, 2020 model; 2022 LiDAR),
        # series B = 2022 anchor (2023, 2024 model; 2025 LiDAR). Model years are estimates.
        for a, (t1, t2), lid, s in ((2018, (2019, 2020), "2018_2022", "A"), (2022, (2023, 2024), "2022_2025", "B")):
            com = base[f"cov{lid}"] & base[f"sc{a}_{t1}"] & base[f"sc{a}_{t2}"]
            gc = base[com].dropna(subset=[key]).groupby(key)
            c[f"{s}_units"] = gc.size().reindex(c.index, fill_value=0)
            for k in ("new", "grew"):
                for t in (t1, t2):
                    c[f"{s}_{k}_{t}"] = (gc[f"M{a}_{t}_{k}"].apply(lambda x: x.fillna(False).astype(bool).sum())
                                         .reindex(c.index, fill_value=0) * fac[k])
                c[f"{s}_{k}_{lid[-4:]}"] = (gc[f"L{lid}_{k}"].apply(lambda x: x.fillna(False).astype(bool).sum())
                                            .reindex(c.index, fill_value=0))
        c["high_rise"] = c.index.isin(flagged[key])
        c.index.name = "cell"
        c.to_csv(OUT / f"cells_{name}.csv")
        for part, cc in (("main", c[~c.high_rise]), ("high-rise", c[c.high_rise])):
            row = {"scale": name, "part": part, "cells": len(cc), "units": int(cc.units.sum())}
            for col in [x for x in c.columns if x.startswith(("lidar_new", "lidar_popup", "lidar_grew", "lidar_demo",
                                                               "model_est", "A_", "B_"))]:
                row[col] = float(cc[col].sum())
            totals.append(row)
        # consistency on the common unit sets (reported only): model t1 <= t2 <= LiDAR at the end of the period
        m = c[~c.high_rise]
        for k in ("new", "grew"):
            for s, ts, end in (("A", (2019, 2020), 2022), ("B", (2023, 2024), 2025)):
                e1, e2, L = m[f"{s}_{k}_{ts[0]}"], m[f"{s}_{k}_{ts[1]}"], m[f"{s}_{k}_{end}"]
                a = 2018 if s == "A" else 2022
                cons.append({"scale": name, "class": k, "anchor": a, "cells": len(m),
                             "share_first_le_second": float((e1 <= e2 + 1e-9).mean()),
                             "share_second_le_lidar": float((e2 <= L + 0.5).mean()),
                             "total_first": float(e1.sum()), "total_second": float(e2.sum()),
                             "total_lidar_period": float(L.sum()),
                             "r_second_vs_lidar": float(np.corrcoef(e2, L)[0, 1]) if e2.std() > 0 and L.std() > 0
                             else np.nan})
    pd.DataFrame(totals).to_csv(OUT / "totals.csv", index=False)
    pd.DataFrame(cons).to_csv(OUT / "consistency.csv", index=False)
    meta = {"thresholds": thr, "calibration_factor": fac, "units_in_mask": int(len(base)),
            "lidar_covered_share": {f"{y0}_{y1}": float(base[f"cov{y0}_{y1}"].mean()) for y0, y1 in PERIODS},
            "model_scored_share": {f"{a}_{t}": float(base[f"sc{a}_{t}"].mean()) for a, t in PAIRS}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    t = pd.DataFrame(totals)
    print(t[(t.scale == "250m") & (t.part == "main")][[c for c in t.columns if c.startswith(("A_", "B_", "lidar_"))]]
          .T.round(0).to_string())
    print(pd.DataFrame(cons).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
