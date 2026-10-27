"""Evaluate a trained height model ONCE on the held-out ZIPs (19146, 19147) against LiDAR.

    python scripts/eval_height.py TAG [--years 2022 2025]

1. Height accuracy per ZIP and year: pixel MAE / R² (LiDAR cells that are not vegetation) and building-level MAE / R²
   (median inside each 2017 footprint, footprints standing in that year), by height class.
2. Change detection 2018→2022 in 19146: the Stage 1 rules (change.Rules, fixed before any test prediction) are run
   on predicted 2018 and 2022 rasters (predicted vegetation masks trees) and compared with the LiDAR flags.
3. Three case figures (crops drawn at random with a fixed seed, not picked by eye).

Writes data/interim/height_pred/{TAG}/*.npz, results/height_model_pilot/metrics_{TAG}.json, metrics.csv (one row per
model / ZIP / year) and figs/case_{TAG}_*.png.
"""

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rasterio import features as rfeat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import height_data as H  # noqa: E402
import height_model as M  # noqa: E402
import tiles as T  # noqa: E402
from match import bootstrap_rate  # noqa: E402
from rasters import Grid  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--years", type=int, nargs="+", default=[2022])
a = ap.parse_args()
RES = ROOT / "results/height_model_pilot"
FIG = RES / "figs"
FIG.mkdir(parents=True, exist_ok=True)
PRED = ROOT / "data/interim/height_pred" / a.tag
PRED.mkdir(parents=True, exist_ok=True)
ck = torch.load(ROOT / f"data/interim/models/height_{a.tag}.pt", map_location="cpu", weights_only=False)
model = M.build(ck["model"], ck["img_res"], random_init=True)  # weights come from the checkpoint
model.load_state_dict(ck["state"])
model = model.cuda().eval()
F_IMG = 2 if ck["img_res"] == 0.25 else 1
CROP, STEP = 256, 128
w1 = np.minimum(np.arange(CROP) + 1, CROP - np.arange(CROP)).astype(np.float32)
WIN = np.outer(w1, w1)  # tent weights: crop centres count more than crop edges


@torch.no_grad()
def predict_patch(img):
    img = M.shift_image(img, *ck["shift"])  # same global registration shift as in training
    if F_IMG == 1:
        img = img.reshape(3, H.P, 2, H.P, 2).mean((2, 4))
    acc = np.zeros((2, H.P, H.P), np.float32)
    wsum = np.zeros((H.P, H.P), np.float32)
    offs = range(0, H.P - CROP + 1, STEP)
    xs, pos = [], []
    for r in offs:
        for c in offs:
            xs.append(img[:, r * F_IMG:(r + CROP) * F_IMG, c * F_IMG:(c + CROP) * F_IMG])
            pos.append((r, c))
    x = torch.from_numpy(np.stack(xs).astype(np.float32) / 255).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x).float()
    h, v = M.split_out(out)
    h, v = h.cpu().numpy(), v.cpu().numpy()
    for k, (r, c) in enumerate(pos):
        acc[0, r:r + CROP, c:c + CROP] += h[k] * WIN
        acc[1, r:r + CROP, c:c + CROP] += v[k] * WIN
        wsum[r:r + CROP, c:c + CROP] += WIN
    return acc / wsum


def predict_raster(zip_code, year):
    """Predicted ndsm / veg on the LiDAR grid of (zip, year); NaN outside the evaluated patches."""
    f = PRED / f"{zip_code}_{year}.npz"
    z = np.load(H.raster_path(zip_code, year))
    grid = Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))
    if f.exists():
        p = np.load(f)
        return {"ndsm": p["ndsm"], "veg": p["veg"]}, z, grid
    nd = np.full((grid.ny, grid.nx), np.nan, np.float32)
    vg = np.full_like(nd, np.nan)
    idx = pd.read_csv(H.OUT / f"{zip_code}_{year}" / "index.csv")
    for i, j in zip(idx.i, idx.j):
        d = np.load(H.OUT / f"{zip_code}_{year}" / f"{i}_{j}.npz")
        hv = predict_patch(d["img"])
        sl = np.s_[i * H.P:(i + 1) * H.P, j * H.P:(j + 1) * H.P]
        nd[sl], vg[sl] = hv[0], hv[1]
    np.savez_compressed(f, ndsm=nd, veg=vg)
    return {"ndsm": nd, "veg": vg}, z, grid


def r2(p, t):
    return float(1 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def height_metrics(zip_code, year, fp):
    pr, z, grid = predict_raster(zip_code, year)
    t, veg = z["ndsm"], z["veg"] > 0.5
    ok = np.isfinite(t) & ~veg & np.isfinite(pr["ndsm"])
    p_, t_ = pr["ndsm"][ok], np.clip(t[ok], 0, M.H_MAX)
    b = t_ > 2.5
    out = {"zip": zip_code, "year": year, "pixels": int(ok.sum()), "pixel_mae": float(np.abs(p_ - t_).mean()),
           "pixel_r2": r2(p_, t_), "bldg_pixel_mae": float(np.abs(p_[b] - t_[b]).mean())}
    # building level: median inside each 2017 footprint (cells that are valid in both maps)
    fps = fp[fp.intersects(T.study_area((zip_code,), buffer_ft=0).union_all())].reset_index(drop=True)
    ids = rfeat.rasterize(zip(fps.geometry.buffer(-0.5), fps.index + 1), out_shape=t.shape, transform=grid.transform,
                          fill=0, dtype="int32")
    m = (ids > 0) & ok
    df = pd.DataFrame({"id": ids[m], "p": pr["ndsm"][m], "t": np.clip(t[m], 0, M.H_MAX)})
    g = df.groupby("id").agg(n=("p", "size"), p=("p", "median"), t=("t", "median"))
    g = g[(g.n >= 20) & (g.t > 2.5)]  # >= 5 m2 measured, building standing in this year
    e = (g.p - g.t).abs().to_numpy()
    out.update({"buildings": int(len(g)), "bldg_mae": float(e.mean()), "bldg_mae_ci": bootstrap_rate(e),
                "bldg_r2": r2(g.p.to_numpy(), g.t.to_numpy()), "bldg_bias": float((g.p - g.t).mean()),
                "bldg_within_1m": float((e <= 1).mean()), "bldg_within_3m": float((e <= 3).mean())})
    for name, lo, hi in [("lt6", 0, 6), ("6to12", 6, 12), ("gt12", 12, 1e9)]:
        s = g[(g.t >= lo) & (g.t < hi)]
        out[f"bldg_mae_{name}"] = float((s.p - s.t).abs().mean()) if len(s) else None
        out[f"n_{name}"] = int(len(s))
    return out, g


def change_metrics():
    """19146, 2018→2022: model-based change flags vs LiDAR-based flags (same rules, same units)."""
    units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
    fp = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                       bbox=tuple(T.study_area().total_bounds))
    pa, _, grid = predict_raster("19146", 2018)
    pb, _, _ = predict_raster("19146", 2022)
    ep = []
    for p in (pa, pb):
        ep.append({"ndsm": p["ndsm"], "veg": np.nan_to_num(p["veg"], nan=0),
                   "ntot": np.isfinite(p["ndsm"]).astype(np.int32)})
    fm = C.unit_features(units, fp, C.Rules(), epochs=(ep[0], ep[1], grid))
    fl = C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), C.Rules())
    m = fl.merge(fm, on="unit", suffixes=("", "_m"))
    m = m[(m.data_18_m > 0.95) & (m.data_22_m > 0.95)]   # units fully inside the predicted area
    res = {"units_evaluated": int(len(m))}
    for k in ("new", "popup"):
        lid, mod = m[f"chg_{k}"].astype(bool), m[f"chg_{k}_m"].astype(bool)
        res[k] = {"lidar_flags": int(lid.sum()), "model_flags": int(mod.sum()),
                  "recall": float((mod & lid).sum() / max(lid.sum(), 1)),
                  "recall_ci": bootstrap_rate(mod[lid].to_numpy(float)),
                  "precision": float((mod & lid).sum() / max(mod.sum(), 1)),
                  "precision_ci": bootstrap_rate(lid[mod].to_numpy(float))}
    lid = m.chg_new | m.chg_popup
    mod = m.chg_new_m | m.chg_popup_m
    res["new_or_popup"] = {"recall": float((mod & lid).sum() / max(lid.sum(), 1)),
                           "precision": float((mod & lid).sum() / max(mod.sum(), 1)),
                           "lidar_flags": int(lid.sum()), "model_flags": int(mod.sum())}
    # dh agreement on LiDAR-flagged pop-ups
    pop = m[m.chg_popup]
    res["popup_dh_p50_lidar_vs_model"] = [float(pop.dh_p50.median()), float(pop.dh_p50_m.median())]
    return res


def cases(n=3, seed=7):
    idx = pd.read_csv(H.OUT / "19146_2022" / "index.csv")
    rng = np.random.default_rng(seed)
    k = 0
    for _ in range(50):
        row = idx.iloc[rng.integers(len(idx))]
        d = np.load(H.OUT / "19146_2022" / f"{int(row.i)}_{int(row.j)}.npz")
        r, c = (int(v) for v in rng.integers(0, H.P - CROP + 1, 2))
        t = d["ndsm"].astype(np.float32)[r:r + CROP, c:c + CROP]
        if np.nanmean(t > 2.5) < 0.2:
            continue  # want crops with buildings in them
        hv = predict_patch(d["img"])[:, r:r + CROP, c:c + CROP]
        img = M.shift_image(d["img"], *ck["shift"])[:, 2 * r:2 * (r + CROP), 2 * c:2 * (c + CROP)].transpose(1, 2, 0)
        fig, ax = plt.subplots(1, 4, figsize=(12, 3.3))
        ax[0].imshow(img)
        ax[0].set_title("aerial 2022")
        im = ax[1].imshow(hv[0], vmin=0, vmax=20, cmap="viridis")
        ax[1].set_title("predicted height")
        ax[2].imshow(np.clip(t, 0, 20), vmin=0, vmax=20, cmap="viridis")
        ax[2].set_title("LiDAR 2022 (nDSM)")
        fig.colorbar(im, ax=ax[1:3], shrink=0.8, label="m")
        er = ax[3].imshow(hv[0] - t, vmin=-6, vmax=6, cmap="RdBu_r")
        ax[3].set_title("predicted − LiDAR")
        fig.colorbar(er, ax=ax[3], shrink=0.8, label="m")
        for q in ax:
            q.set_axis_off()
        fig.savefig(FIG / f"case_{a.tag}_{k + 1}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        k += 1
        if k == n:
            break


fp_all = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                       bbox=tuple(T.study_area(M.TEST_ZIPS).total_bounds))
out = {"tag": a.tag, "model": ck["model"], "img_res": ck["img_res"], "ckpt_iter": ck["iter"],
       "val_bldg_pixel_mae": ck["val_bldg_mae"], "height": []}
rows = []
for zc in M.TEST_ZIPS:
    for y in a.years:
        if not (H.OUT / f"{zc}_{y}" / "index.csv").exists():
            print(f"skip {zc} {y}: no patches")
            continue
        h, _ = height_metrics(zc, y, fp_all)
        out["height"].append(h)
        rows.append({"tag": a.tag, **{k: v for k, v in h.items() if not isinstance(v, (list, tuple))}})
        print(json.dumps(h), flush=True)
if (H.OUT / "19146_2018" / "index.csv").exists() and 2022 in a.years:
    out["change_19146_2018_2022"] = change_metrics()
    print(json.dumps(out["change_19146_2018_2022"], indent=1))
M.save_json(out, RES / f"metrics_{a.tag}.json")
cases()
mc = RES / "metrics.csv"
old = pd.read_csv(mc) if mc.exists() else pd.DataFrame()
if len(old):
    old = old[old.tag != a.tag]
pd.concat([old, pd.DataFrame(rows)]).to_csv(mc, index=False)
print("written", RES / f"metrics_{a.tag}.json")
