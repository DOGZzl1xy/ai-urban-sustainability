"""Grid-level evaluation of the unit classifier on the development test (19146 + 19147, 2018 -> 2022).

    python scripts/grid_eval.py TAG [TAG ...]

For grew (new construction or pop-up), new construction and pop-up, units are counted per 100 m cell, 250 m cell and
2019 census block group (unit centroids). The model count is calibrated with the VALIDATION precision and recall of
the same checkpoint and thresholds: estimate = flagged x precision / recall. Compared with the LiDAR count per cell:
Pearson r, mean absolute error and mean error (bias), for raw flags and calibrated estimates.
Inputs: data/interim/unitcls_preds/{TAG}_dev_2018_2022.parquet and {TAG}_val_2018_2022.parquet (written by
scripts/unitcls.py dev). Outputs: results/bitemporal/grid_eval_{TAG}.csv / .json and figs/grid_eval_{TAG}.png.
"""

import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_model as M  # noqa: E402
from rasters import FT  # noqa: E402
from viz import PALETTE, style  # noqa: E402

PRED = ROOT / "data/interim/unitcls_preds"
RES = ROOT / "results/bitemporal"
(RES / "figs").mkdir(parents=True, exist_ok=True)
CLASSES = ("grew", "new", "popup")


def flags(df, zips, period="2018_2022"):
    """Model flags and LiDAR truth for the three classes; truth for 'new' from the Stage 1 LiDAR flags."""
    d = df.copy()
    d["m_grew"] = d.logit_grew >= d.thr_grew
    d["m_popup"] = d.logit_popup >= d.thr_popup
    d["m_new"] = d.m_grew & ~d.m_popup
    new = pd.concat([pd.read_parquet(ROOT / f"data/interim/lidar_change/{z}_{period}.parquet")[["unit", "chg_new"]]
                     .assign(zip=z) for z in zips])
    d = d.merge(new, on=["zip", "unit"], how="left")
    d["t_grew"], d["t_popup"], d["t_new"] = d.grew.astype(bool), d.popup.astype(bool), d.chg_new.fillna(False)
    return d


def calibration(v):
    out = {}
    for k in CLASSES:
        m, t = v[f"m_{k}"], v[f"t_{k}"]
        tp = int((m & t).sum())
        p, r = tp / max(m.sum(), 1), tp / max(t.sum(), 1)
        out[k] = {"precision": p, "recall": r, "factor": p / r if r > 0 else np.nan, "val_flags": int(m.sum()),
                  "val_truth": int(t.sum())}
    return out


def cells(d):
    """Assign unit centroids to 100 m and 250 m cells and to 2019 census block groups."""
    units = pd.concat([gpd.read_parquet(ROOT / f"data/interim/units_{z}.parquet")[["unit", "geometry"]].assign(zip=z)
                       for z in d.zip.unique()])
    g = gpd.GeoDataFrame(d.merge(units, on=["zip", "unit"]), geometry="geometry", crs=2272)
    c = g.geometry.centroid
    for m in (100, 250):
        s = m / FT
        g[f"c{m}"] = [f"{int(x // s)}_{int(y // s)}" for x, y in zip(c.x, c.y)]
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2019_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    bg = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2019_42_bg.zip'}!{name}")
    bg = bg[bg.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]]
    pts = gpd.GeoDataFrame(g[["zip", "unit"]], geometry=c, crs=2272)
    j = gpd.sjoin(pts, bg, how="left", predicate="within")
    g["bg"] = j.GEOID.to_numpy()
    return g


def evaluate(tag, final=False):
    period = "2022_2025" if final else "2018_2022"
    dev = pd.read_parquet(PRED / f"{tag}_{'FINAL' if final else 'dev'}_{period}.parquet")
    val = pd.read_parquet(PRED / f"{tag}_val_2018_2022.parquet")
    d = flags(dev, M.TEST_ZIPS, period)
    tag_out = f"{tag}_FINAL" if final else tag
    cal = calibration(flags(val, val.zip.unique()))
    g = cells(d)
    rows = []
    fig, axes = plt.subplots(3, 3, figsize=(11, 10.5))
    for i, scale in enumerate(("c100", "c250", "bg")):
        agg = g.dropna(subset=[scale]).groupby(scale).agg(
            units=("unit", "size"), **{f"{p}_{k}": (f"{p}_{k}", "sum") for k in CLASSES for p in ("m", "t")})
        for j, k in enumerate(CLASSES):
            f = cal[k]["factor"]
            est = agg[f"m_{k}"] * f
            tru = agg[f"t_{k}"]
            r = float(np.corrcoef(est, tru)[0, 1]) if tru.std() > 0 and est.std() > 0 else np.nan
            rows.append({"tag": tag_out, "scale": {"c100": "100 m", "c250": "250 m", "bg": "block group"}[scale],
                         "class": k, "cells": int(len(agg)), "lidar_total": int(tru.sum()),
                         "flagged_total": int(agg[f"m_{k}"].sum()), "estimate_total": float(est.sum()),
                         "calibration_factor": f, "r_raw": float(np.corrcoef(agg[f"m_{k}"], tru)[0, 1])
                         if agg[f"m_{k}"].std() > 0 and tru.std() > 0 else np.nan,
                         "r_calibrated": r, "mae": float((est - tru).abs().mean()), "mean_error": float((est - tru).mean())})
            ax = axes[i, j]
            lim = max(tru.max(), est.max(), 1) * 1.05
            ax.plot([0, lim], [0, lim], color="0.7", lw=0.8)
            ax.scatter(tru, est, s=10, alpha=0.6, color=PALETTE["primary"], lw=0)
            ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="LiDAR count", ylabel="model estimate")
            ax.set_title(f"{k} · {rows[-1]['scale']} · r = {r:.2f}", fontsize=9)
    fig.suptitle(f"{tag}: calibrated model counts vs LiDAR, "
                 f"{'FINAL test 2022→2025' if final else 'development test 2018→2022'}", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / f"figs/grid_eval_{tag_out}.png", dpi=150)
    plt.close(fig)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"grid_eval_{tag_out}.csv", index=False)
    M.save_json({"tag": tag_out, "calibration_from_validation": cal, "metrics": out.to_dict("records")},
                RES / f"grid_eval_{tag_out}.json")
    print(out[["scale", "class", "cells", "lidar_total", "estimate_total", "r_raw", "r_calibrated", "mae",
               "mean_error"]].round(2).to_string(index=False))
    return out


if __name__ == "__main__":
    style()
    fin = "--final" in sys.argv
    for t in [x for x in sys.argv[1:] if x != "--final"]:
        print(f"== {t}{' (FINAL 2022-2025)' if fin else ''}")
        evaluate(t, fin)
    print(json.dumps({"done": sys.argv[1:]}))
