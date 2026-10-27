"""Round-2 final test (results/citywide/M4_plan.md): the eight sealed test ZIPs, 2022 -> 2025, run ONCE.

    POPUP_DATA=citywide python scripts/m4_test.py select          # test patch index (rules of m4_patches.py); counts
    POPUP_DATA=citywide python scripts/m4_test.py cut [--threads 8]
    POPUP_DATA=citywide python scripts/m4_test.py prepare         # unit-id patches + pair list (no labels printed)
    POPUP_DATA=citywide python scripts/m4_test.py run             # ONCE: score both models, write all results

Patches: 512-cell patches of each test ZIP's window, LiDAR height on >= 80% of cells, >= 50% inside the ZIP, >= 25%
inside the study-area mask (the train / val rules, without a split), years 2022 and 2025; pairs = patches present
and not blank in both years. Units are scored as in validation (scripts/unitcls.py score_units: 256-cell windows at
stride 128, unit >= 90% inside a window, covered by LiDAR in both years; mean logit over windows).
Models: final M4 (`m4s_s0`: pop-up head unitcls_m4s_s0.pt, grew head unitcls_m4s_s0_grew.pt) and M3b seed 1
(`v3b_s1`, frozen), each with its own validation thresholds and count calibration (precision / recall at those
thresholds on its own validation units: citywide validation for M4, Stage 3 validation for M3b).
High-rise rule (study_area.md): 100 m / 250 m cells and block groups flagged in results/citywide/highrise_all.csv
are left out of the main grid results and reported separately; units in a flagged 100 m cell are left out of the
main unit-level results. Inference model = higher calibrated pop-up r at 250 m (main cells); tie < 0.01 -> block-
group pop-up r. Intermediate-year pop-ups only if that r >= 0.8.
Outputs: results/citywide/m4/round2_test.json, round2_grid.csv, data/interim/unitcls_preds/{tag}_ROUND2_2022_2025.parquet
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from rasterio import features as rfeat
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))  # after src/: "unitcls" must be src/unitcls.py, not the training script
import bitemporal as B  # noqa: E402
import height_data as H  # noqa: E402
import height_model as M  # noqa: E402
import tiles as T  # noqa: E402
import unitcls as U  # noqa: E402
from match import bootstrap_rate  # noqa: E402
from rasters import FT, Grid  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
DRY = "--dry" in sys.argv  # code check on two train / val ZIPs, 2018 -> 2022, 40 pairs: no lock, no test data
Y0, Y1 = (2018, 2022) if DRY else (2022, 2025)
CROP = 256
RES = ROOT / "results/citywide/m4"
PRED = ROOT / "data/interim/unitcls_preds"
IDX = ROOT / "data/interim/m4_test_patches.csv"
LOCK = RES / "round2_test_log.csv"
SFX = "_DRY" if DRY else ""
MODELS = {"m4s_s0": {"popup": "unitcls_m4s_s0.pt", "grew": "unitcls_m4s_s0_grew.pt",
                     "val": "m4s_s0_valbest_2018_2022.parquet", "val_change": "lidar_change_city"},
          "v3b_s1": {"popup": "unitcls_v3b_s1.pt", "grew": "unitcls_v3b_s1_grew.pt",
                     "val": "v3b_s1_val_2018_2022.parquet", "val_change": "lidar_change"}}
CLASSES = ("grew", "new", "popup")
zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
TEST = ("19146", "19147") if DRY else tuple(sorted(zt[zt.test].zip))


def select():
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
    rows = []
    for z in TEST:
        zarea = T.study_area((z,), buffer_ft=0).union_all()
        for y in (Y0, Y1):
            d = np.load(ROOT / f"data/interim/citywide_rasters/{z}_{y}.npz")
            g = Grid(float(d["x0"]), float(d["y1"]), int(d["nx"]), int(d["ny"]))
            nd = d["ndsm"]
            inm = rfeat.rasterize([(zarea.intersection(mask), 1)], out_shape=(g.ny, g.nx), transform=g.transform,
                                  fill=0, dtype="uint8")
            for p in H.patches(g.x0, g.y1, g.nx, g.ny):
                sl = np.s_[p.i * H.P:(p.i + 1) * H.P, p.j * H.P:(p.j + 1) * H.P]
                if inm[sl].mean() < 0.25:
                    continue
                valid = float(np.isfinite(nd[sl]).mean())
                b = box(*p.bounds)
                inside = b.intersection(zarea).area / b.area
                if valid < 0.8 or inside < 0.5:
                    continue
                rows.append(dict(zip=z, year=y, i=p.i, j=p.j, x0=p.x0, y1=p.y1, valid=round(valid, 3),
                                 inside=round(inside, 3), in_mask=round(float(inm[sl].mean()), 3), split="test"))
    idx = pd.DataFrame(rows)
    idx.to_csv(IDX, index=False)
    both = idx[idx.year == Y0].merge(idx[idx.year == Y1][["zip", "i", "j"]], on=["zip", "i", "j"])
    print(f"test patches: {len(idx[idx.year == Y0])} ({Y0}), {len(idx[idx.year == Y1])} ({Y1}); "
          f"in both years {len(both)}", flush=True)


def cut(threads):
    import m4_cut
    idx = pd.read_csv(IDX, dtype={"zip": str})
    for (z, y), sel in idx.groupby(["zip", "year"]):
        t = time.time()
        m4_cut.cut(z, int(y), sel, threads)
        print(f"  cut {z} {y}: {len(sel)} patches ({time.time() - t:.0f}s)", flush=True)
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts/unpack_patches.py"),
                    *[f"{z}_{y}" for z, y in idx[["zip", "year"]].drop_duplicates().itertuples(index=False)]],
                   check=True)


def pairs_units():
    """Test pairs with unit-id paths, unit sizes and labels, keyed by (zip, Y0, Y1) as in scripts/unitcls.py."""
    pi = B.pair_index(TEST, Y0, Y1)
    if DRY:
        pi = pi.head(40).reset_index(drop=True)
    sizes, labels = {}, {}
    for z in TEST:
        U.unit_id_patches(z, Y1)
        sizes[(z, Y0, Y1)] = U.unit_id_patches(z, Y0)
        labels[(z, Y0, Y1)] = U.unit_labels(z, Y0, Y1)
    pi["uid"] = [str(U.UID_DIR / f"{z}_{i}_{j}.npy") for z, i, j in zip(pi.zip, pi.i, pi.j)]
    pi["key"] = [(z, Y0, Y1) for z in pi.zip]
    return pi, sizes, labels


@torch.no_grad()
def score_units(model, pi, sizes, labels, shift):
    """Copy of scripts/unitcls.py score_units (same windows, same unit rule)."""
    model.eval()
    acc, first = {}, {}
    offs = range(0, H.P - CROP + 1, 128)
    for z, i, j, p0, p1, up in zip(pi.key, pi.i, pi.j, pi.path0, pi.path1, pi.uid):
        a0, b0 = B.load_patch(p0, filled=False), B.load_patch(p1, filled=False)
        im0, im1 = M.shift_image(a0["img"], *shift), M.shift_image(b0["img"], *shift)
        h0, g0, ud = np.asarray(a0["ndsm"], np.float32), np.asarray(a0["veg"]), np.load(up)
        pos = [(r, c) for r in offs for c in offs]
        x0 = torch.from_numpy(np.stack([im0[:, 2 * r:2 * (r + CROP), 2 * c:2 * (c + CROP)] for r, c in pos])
                              .astype(np.float32) / 255).cuda()
        x1 = torch.from_numpy(np.stack([im1[:, 2 * r:2 * (r + CROP), 2 * c:2 * (c + CROP)] for r, c in pos])
                              .astype(np.float32) / 255).cuda()
        hh = torch.from_numpy(np.stack([np.clip(np.nan_to_num(h0[r:r + CROP, c:c + CROP]), 0, M.H_MAX)
                                        for r, c in pos])).cuda()
        gg = torch.from_numpy(np.stack([g0[r:r + CROP, c:c + CROP] for r, c in pos])).cuda()
        uu = torch.from_numpy(np.stack([ud[r:r + CROP, c:c + CROP] for r, c in pos]).astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, (logits, uniq, n) = model(x0, x1, hh, gg, uu)
        logits, uniq, n = logits.float().cpu().numpy(), uniq.cpu().numpy(), n.cpu().numpy()
        uid = uniq % 10_000_000 - 1
        for q in range(len(uniq)):
            tot = sizes[z].get(uid[q], 0)
            if tot and n[q] >= U.MIN_IN_CROP * tot:
                acc.setdefault((z, int(uid[q])), []).append(logits[q])
                first.setdefault((z, int(uid[q])), (i, j))
    rows = []
    for (z, u), ls in acc.items():
        if u in labels[z].index and labels[z].loc[u].covered:
            m = np.mean(ls, 0)
            rows.append({"zip": z[0], "unit": u, "logit_grew": m[0], "logit_popup": m[1],
                         "grew": labels[z].loc[u].grew, "popup": labels[z].loc[u].popup,
                         "i": first[(z, u)][0], "j": first[(z, u)][1]})
    return pd.DataFrame(rows)


def load_model(path):
    ck = torch.load(ROOT / "data/interim/models" / path, map_location="cpu", weights_only=False)
    m = U.UnitCls(random_init=True, pool=ck.get("pool", "mean"))
    m.load_state_dict(ck["state"])
    return m.cuda(), ck


def ap_score(y, s):
    o = np.argsort(-s, kind="stable")
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


def with_truth(df, change_dir, y0, y1):
    """Add the 'new' truth (LiDAR new construction) next to grew / pop-up."""
    new = pd.concat([pd.read_parquet(ROOT / f"data/interim/{change_dir}/{z}_{y0}_{y1}.parquet")[["unit", "chg_new"]]
                     .assign(zip=z) for z in df.zip.unique()])
    d = df.merge(new, on=["zip", "unit"], how="left")
    d["t_grew"], d["t_popup"], d["t_new"] = d.grew.astype(bool), d.popup.astype(bool), d.chg_new.fillna(False)
    return d


def model_flags(d, thr):
    d["m_grew"] = d.logit_grew >= thr["grew"]
    d["m_popup"] = d.logit_popup >= thr["popup"]
    d["m_new"] = d.m_grew & ~d.m_popup
    return d


def unit_report(d):
    out = {"units": int(len(d))}
    for k in ("grew", "popup"):
        y, s, m = d[f"t_{k}"].to_numpy(), d[f"logit_{k}"].to_numpy(), d[f"m_{k}"].to_numpy()
        tp = int((m & y).sum())
        out[k] = {"lidar": int(y.sum()), "model": int(m.sum()), "ap": ap_score(y.astype(float), s),
                  "recall": tp / max(y.sum(), 1), "precision": tp / max(m.sum(), 1),
                  "recall_ci": bootstrap_rate(m[y].astype(float)), "precision_ci": bootstrap_rate(y[m].astype(float))}
    return out


def geography(d):
    units = pd.concat([gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")[
        ["unit", "geometry", "area_type", "in_mask"]].assign(zip=z) for z in d.zip.unique()])
    g = gpd.GeoDataFrame(d.merge(units, on=["zip", "unit"]), geometry="geometry", crs=2272)
    c = g.geometry.centroid
    for m in (100, 250):
        s = m / FT
        g[f"c{m}"] = [f"{int(x // s)}_{int(y // s)}" for x, y in zip(c.x, c.y)]
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2019_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    bg = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2019_42_bg.zip'}!{name}")
    bg = bg[bg.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]]
    j = gpd.sjoin(gpd.GeoDataFrame(g[["zip", "unit"]], geometry=c, crs=2272), bg, how="left", predicate="within")
    g["bg"] = j.GEOID.to_numpy()
    return pd.DataFrame(g.drop(columns="geometry"))


def grid_rows(tag, g, cal, flagged, part):
    rows = []
    for scale, name in (("c100", "100 m"), ("c250", "250 m"), ("bg", "block group")):
        h = g.dropna(subset=[scale])
        h = h[h[scale].isin(flagged[scale])] if part == "high-rise" else h[~h[scale].isin(flagged[scale])]
        agg = h.groupby(scale).agg(**{f"{p}_{k}": (f"{p}_{k}", "sum") for k in CLASSES for p in ("m", "t")})
        for k in CLASSES:
            est, tru, raw = agg[f"m_{k}"] * cal[k]["factor"], agg[f"t_{k}"], agg[f"m_{k}"]
            ok = len(agg) > 2 and tru.std() > 0
            rows.append({"tag": tag, "part": part, "scale": name, "class": k, "cells": int(len(agg)),
                         "lidar_total": int(tru.sum()), "flagged_total": int(raw.sum()),
                         "estimate_total": float(est.sum()), "calibration_factor": cal[k]["factor"],
                         "r_raw": float(np.corrcoef(raw, tru)[0, 1]) if ok and raw.std() > 0 else np.nan,
                         "r_calibrated": float(np.corrcoef(est, tru)[0, 1]) if ok and est.std() > 0 else np.nan,
                         "mae": float((est - tru).abs().mean()) if len(agg) else np.nan,
                         "mean_error": float((est - tru).mean()) if len(agg) else np.nan})
    return rows


def run():
    if not DRY:
        if LOCK.exists():
            sys.exit(f"the round-2 test was already run: {LOCK.read_text()}")
        LOCK.write_text(f"models,time\n{'+'.join(MODELS)},{pd.Timestamp.now().isoformat()}\n")  # before scoring
    pi, sizes, labels = pairs_units()
    hr = pd.read_csv(ROOT / f"results/citywide/highrise_{'trainval' if DRY else 'all'}.csv", dtype={"key": str})
    flagged = {u: set(hr[(hr.unit == u) & hr.flagged].key) for u in ("c100", "c250", "bg")}
    zone = dict(zip(zt.zip, zt.type))
    res, grid = {"test_zips": TEST, "period": f"{Y0}-{Y1}", "pairs": int(len(pi)), "models": {}}, []
    for tag, f in MODELS.items():
        mp, ckp = load_model(f["popup"])
        df = score_units(mp, pi, sizes, labels, tuple(ckp["shift"]))
        del mp
        mg, ckg = load_model(f["grew"])
        dg = score_units(mg, pi, sizes, labels, tuple(ckg["shift"]))
        del mg
        torch.cuda.empty_cache()
        df = df.drop(columns="logit_grew").merge(dg[["zip", "unit", "logit_grew"]], on=["zip", "unit"])
        thr = {"popup": ckp["thresholds"]["popup"], "grew": ckg["thresholds"]["grew"]}
        df.assign(thr_grew=thr["grew"], thr_popup=thr["popup"]).to_parquet(PRED / f"{tag}_ROUND2{SFX}_{Y0}_{Y1}.parquet")
        # count calibration from the model's own validation units and thresholds
        v = model_flags(with_truth(pd.read_parquet(PRED / f["val"]), f["val_change"], 2018, 2022), thr)
        cal = {}
        for k in CLASSES:
            tp = int((v[f"m_{k}"] & v[f"t_{k}"]).sum())
            p, r = tp / max(v[f"m_{k}"].sum(), 1), tp / max(v[f"t_{k}"].sum(), 1)
            cal[k] = {"precision": p, "recall": r, "factor": p / r if r > 0 else np.nan}
        d = geography(model_flags(with_truth(df, "lidar_change_city", Y0, Y1), thr))
        main = d[~d.c100.isin(flagged["c100"])]
        res["models"][tag] = {
            "checkpoints": {"popup": [f["popup"], int(ckp["iter"])], "grew": [f["grew"], int(ckg["iter"])]},
            "thresholds_from_validation": thr, "calibration_from_validation": cal,
            "units_main": unit_report(main), "units_high_rise": unit_report(d[d.c100.isin(flagged["c100"])]),
            "units_by_unit_area_type": {t: unit_report(x) for t, x in main.groupby("area_type")},
            "units_by_zip_type": {t: unit_report(x) for t, x in main.assign(zt=main.zip.map(zone)).groupby("zt")},
            "flag_rate": {k: float(d[f"m_{k}"].mean()) for k in CLASSES}}
        grid += grid_rows(tag, d, cal, flagged, "main") + grid_rows(tag, d, cal, flagged, "high-rise")
    gr = pd.DataFrame(grid)
    gr.to_csv(RES / f"round2_grid{SFX}.csv", index=False)
    # positive rates (LiDAR) of both periods in the test ZIPs, all covered units in the study-area mask
    rates = {}
    for y0, y1 in ((2018, 2022), (2022, 2025)):
        lab = pd.concat([U.unit_labels(z, y0, y1).assign(zip=z) for z in TEST])
        inm = pd.concat([gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")[["unit", "in_mask"]]
                         .assign(zip=z) for z in TEST])
        lab = lab.reset_index().merge(inm, on=["zip", "unit"])
        lab = lab[lab.covered & lab.in_mask]
        rates[f"{y0}-{y1}"] = {"units": int(len(lab)), "grew": float(lab.grew.mean()), "popup": float(lab.popup.mean())}
    res["lidar_positive_rates_test_zips"] = rates
    m250 = gr[(gr.part == "main") & (gr["class"] == "popup") & (gr.scale == "250 m")].set_index("tag").r_calibrated
    mbg = gr[(gr.part == "main") & (gr["class"] == "popup") & (gr.scale == "block group")].set_index("tag").r_calibrated
    a_, b_ = list(MODELS)
    pick = (a_ if m250[a_] > m250[b_] else b_) if abs(m250[a_] - m250[b_]) >= 0.01 else (a_ if mbg[a_] >= mbg[b_] else b_)
    res["inference_model"] = {"tag": pick, "popup_r_250m": float(m250[pick]),
                              "intermediate_year_popups_allowed": bool(m250[pick] >= 0.8)}
    M.save_json(res, RES / f"round2_test{SFX}.json")
    print(json.dumps({"inference_model": res["inference_model"], "rates": rates}, indent=1, default=float))
    print(gr[gr.part == "main"][["tag", "scale", "class", "cells", "lidar_total", "estimate_total", "r_calibrated",
                                 "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "cut", "prepare", "run"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.step == "select":
        select()
    elif a.step == "cut":
        cut(a.threads)
    elif a.step == "prepare":
        pi_, _, _ = pairs_units()
        print(f"prepared {len(pi_)} test pairs ({Y0}-{Y1})", flush=True)
    else:
        run()
