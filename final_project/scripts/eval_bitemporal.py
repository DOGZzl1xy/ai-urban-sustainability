"""Evaluate bi-temporal models (and the single-image baseline) on vertical change.

    python scripts/eval_bitemporal.py MODEL [MODEL ...]            # development test: 2018 -> 2022 only
    python scripts/eval_bitemporal.py --final MODEL               # FINAL test 2022 -> 2025: run once, frozen model
        MODEL = bitemp_<tag> (data/interim/models/bitemp_<tag>.pt) or height_<tag> (single-image, e.g. height_dino_dpt_r025)

Protocol (every model the same way):
  "before" = LiDAR of the earlier year; "after" = the model's height for the later year (single-image: from the later
  image only; bi-temporal: from both images, plus the LiDAR anchor for variant B). The Stage 1 rules turn before /
  after into change flags per unit, compared with the LiDAR flags on units fully covered by both.
  1. Pop-up thresholds (up_h, up_share) are tuned per model on the Stage 3 VALIDATION blocks of the training
     ZIPs (B.make_splits_bt; 2018 -> 2022, max F1 against LiDAR); everything else keeps the Stage 1 values.
     Note: the single-image baseline was trained on the Stage 2 split, so some Stage 3 validation patches were in
     its training data; this can only flatter the baseline.
  2. Test ZIPs 19146 and 19147. 2018 -> 2022 (held out in space) is the DEVELOPMENT test, open to every model
     version. 2022 -> 2025 (held out in time and space) is SEALED: it runs only with --final, for the one model
     version chosen before looking at it, and each --final run is logged in results/bitemporal/final_test_log.csv.
     Reported with tuned and with default thresholds.
Writes results/bitemporal/eval_<MODEL>.json and eval_summary.csv.
"""

import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from rasterio import features as rfeat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bitemporal as B  # noqa: E402
import change as C  # noqa: E402
import height_data as H  # noqa: E402
import height_model as M  # noqa: E402
import tiles as T  # noqa: E402
from match import bootstrap_rate  # noqa: E402
from rasters import Grid  # noqa: E402

RES = ROOT / "results/bitemporal"
RES.mkdir(parents=True, exist_ok=True)
CROP, STEP = 256, 128
w1 = np.minimum(np.arange(CROP) + 1, CROP - np.arange(CROP)).astype(np.float32)
WIN = np.outer(w1, w1)
FP = gpd.read_file(ROOT / "data/raw/footprints/PhiladelphiaBuildings2017/PhiladelphiaBuildings2017.shp",
                   bbox=tuple(T.study_area(M.TRAIN_ZIPS + M.TEST_ZIPS).total_bounds))


class Predictor:
    def __init__(self, name):
        self.name = name
        if name.startswith("bitemp_"):
            ck = torch.load(ROOT / f"data/interim/models/{name}.pt", map_location="cpu", weights_only=False)
            self.kind, self.variant = "bitemp", ck["variant"]
            self.model = B.BiTemporal(ck["variant"], random_init=True)
        else:
            ck = torch.load(ROOT / f"data/interim/models/{name}.pt", map_location="cpu", weights_only=False)
            self.kind, self.variant = "single", None
            self.model = M.build(ck["model"], ck["img_res"], random_init=True)
        self.model.load_state_dict(ck["state"])
        self.model = self.model.cuda().eval()
        self.shift = tuple(ck["shift"])
        self.out_dir = ROOT / "data/interim/bitemporal_pred" / name
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def patch(self, img0, img1, h0, g0):
        """Predicted later-year height and vegetation probability on one 512 x 512 patch."""
        img0, img1 = M.shift_image(img0, *self.shift), M.shift_image(img1, *self.shift)
        acc = np.zeros((2, H.P, H.P), np.float32)
        ws = np.zeros((H.P, H.P), np.float32)
        pos = [(r, c) for r in range(0, H.P - CROP + 1, STEP) for c in range(0, H.P - CROP + 1, STEP)]
        cut = lambda im, r, c: im[:, 2 * r:2 * (r + CROP), 2 * c:2 * (c + CROP)]  # noqa: E731
        x0 = torch.from_numpy(np.stack([cut(img0, r, c) for r, c in pos]).astype(np.float32) / 255).cuda()
        x1 = torch.from_numpy(np.stack([cut(img1, r, c) for r, c in pos]).astype(np.float32) / 255).cuda()
        a0 = torch.from_numpy(np.stack([np.clip(np.nan_to_num(h0[r:r + CROP, c:c + CROP]), 0, M.H_MAX)
                                        for r, c in pos])).cuda()
        v0 = torch.from_numpy(np.stack([g0[r:r + CROP, c:c + CROP] for r, c in pos])).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = (self.model(x0, x1, a0, v0) if self.kind == "bitemp" else self.model(x1)).float()
        hh = out[:, 0] + a0 if self.variant == "pair" else out[:, 0]
        vv = torch.sigmoid(out[:, 1])
        hh, vv = hh.cpu().numpy(), vv.cpu().numpy()
        for k, (r, c) in enumerate(pos):
            acc[0, r:r + CROP, c:c + CROP] += hh[k] * WIN
            acc[1, r:r + CROP, c:c + CROP] += vv[k] * WIN
            ws[r:r + CROP, c:c + CROP] += WIN
        return acc / ws

    def raster(self, zip_code, y0, y1, only=None):
        """Predicted y1 rasters on the LiDAR grid (NaN outside predicted patches). only: set of (i, j) to predict."""
        f = self.out_dir / f"{zip_code}_{y0}_{y1}.npz"
        if f.exists() and only is None:
            d = np.load(f)
            return d["ndsm"], d["veg"]
        z0 = np.load(H.raster_path(zip_code, y0))
        nd = np.full(z0["ndsm"].shape, np.nan, np.float32)
        vg = np.full_like(nd, np.nan)
        pi = B.pair_index((zip_code,), y0, y1)
        for i, j, p0, p1 in zip(pi.i, pi.j, pi.path0, pi.path1):
            if only is not None and (i, j) not in only:
                continue
            a, b = np.load(p0), np.load(p1)
            hv = self.patch(a["img"], b["img"], a["ndsm"].astype(np.float32), a["veg"])
            sl = np.s_[i * H.P:(i + 1) * H.P, j * H.P:(j + 1) * H.P]
            nd[sl], vg[sl] = hv[0], hv[1]
        if only is None:
            np.savez_compressed(f, ndsm=nd, veg=vg)
        return nd, vg


def lidar_epoch(zip_code, year):
    z = np.load(H.raster_path(zip_code, year))
    return {k: z[k] for k in ("ndsm", "veg", "ntot")}, Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))


def unit_flags(pred, zip_code, y0, y1, rules, units=None):
    """Stage 1 features with LiDAR y0 as before and the prediction as after."""
    a, grid = lidar_epoch(zip_code, y0)
    nd, vg = pred
    b = {"ndsm": nd, "veg": np.nan_to_num(vg, nan=0), "ntot": np.isfinite(nd).astype(np.int32)}
    units = units if units is not None else gpd.read_parquet(ROOT / f"data/interim/units_{zip_code}.parquet")
    fp = FP[FP.intersects(T.study_area((zip_code,)).union_all())]
    return C.unit_features(units, fp, rules, epochs=(a, b, grid))


def compare(fm, zip_code, y0, y1):
    """fm: model-based unit flags; reference: LiDAR flags with Stage 1 rules. Units fully covered by both."""
    fl = pd.read_parquet(ROOT / f"data/interim/lidar_change/{zip_code}_{y0}_{y1}.parquet")
    m = fl.merge(fm, on="unit", suffixes=("", "_m"))
    m = m[(m.data_18 > 0.95) & (m.data_22 > 0.95) & (m.data_22_m > 0.95)]
    res = {"units": int(len(m))}
    for k in ("new", "popup"):
        lid, mod = m[f"chg_{k}"].astype(bool), m[f"chg_{k}_m"].astype(bool)
        tp = int((lid & mod).sum())
        rec, prec = tp / max(lid.sum(), 1), tp / max(mod.sum(), 1)
        res[k] = {"lidar": int(lid.sum()), "model": int(mod.sum()), "recall": rec, "precision": prec,
                  "f1": 2 * rec * prec / max(rec + prec, 1e-9),
                  "recall_ci": bootstrap_rate(mod[lid].to_numpy(float)),
                  "precision_ci": bootstrap_rate(lid[mod].to_numpy(float))}
    pop = m[m.chg_popup]
    res["popup_rise_median_lidar_vs_model"] = [float(pop.dh_p50.median()), float(pop.dh_p50_m.median())]
    return res


def height_mae(pred, zip_code, y1):
    z = np.load(H.raster_path(zip_code, y1))
    grid = Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))
    t, ok = z["ndsm"], np.isfinite(z["ndsm"]) & ~(z["veg"] > 0.5) & np.isfinite(pred[0])
    fps = FP[FP.intersects(T.study_area((zip_code,), buffer_ft=0).union_all())].reset_index(drop=True)
    ids = rfeat.rasterize(zip(fps.geometry.buffer(-0.5), fps.index + 1), out_shape=t.shape,
                          transform=grid.transform, fill=0, dtype="int32")
    mk = (ids > 0) & ok
    g = pd.DataFrame({"id": ids[mk], "p": pred[0][mk], "t": np.clip(t[mk], 0, M.H_MAX)}).groupby("id").agg(
        n=("p", "size"), p=("p", "median"), t=("t", "median"))
    g = g[(g.n >= 20) & (g.t > 2.5)]
    e = (g.p - g.t).abs().to_numpy()
    return {"buildings": int(len(g)), "bldg_mae": float(e.mean()), "bldg_mae_ci": bootstrap_rate(e)}


def tune(pr):
    """Pop-up thresholds maximising F1 on validation-block units of the training ZIPs (2018 -> 2022)."""
    sp = B.make_splits_bt()
    rows = []
    cache = {}
    for z in M.TRAIN_ZIPS:
        val = sp[(sp.zip == z) & (sp.split == "val")]
        if not len(val):
            continue
        only = set(zip(val.i, val.j))
        cache[z] = pr.raster(z, 2018, 2022, only=only)
    grid_h, grid_s = (1.5, 2.0, 2.5, 3.0), (0.2, 0.3, 0.4, 0.5)
    for up_h, up_s in itertools.product(grid_h, grid_s):
        rules = replace(C.Rules(), up_h=up_h, up_share=up_s)
        tp = nl = nm = 0
        for z, pred in cache.items():
            fm = unit_flags(pred, z, 2018, 2022, rules)
            fl = pd.read_parquet(ROOT / f"data/interim/lidar_change/{z}_2018_2022.parquet")
            m = fl.merge(fm, on="unit", suffixes=("", "_m"))
            m = m[(m.data_18 > 0.95) & (m.data_22 > 0.95) & (m.data_22_m > 0.95)]
            lid, mod = m.chg_popup.astype(bool), m.chg_popup_m.astype(bool)
            tp, nl, nm = tp + int((lid & mod).sum()), nl + int(lid.sum()), nm + int(mod.sum())
        rec, prec = tp / max(nl, 1), tp / max(nm, 1)
        rows.append({"up_h": up_h, "up_share": up_s, "val_popups": nl, "recall": rec, "precision": prec,
                     "f1": 2 * rec * prec / max(rec + prec, 1e-9)})
    t = pd.DataFrame(rows).sort_values("f1", ascending=False)
    best = t.iloc[0]
    return replace(C.Rules(), up_h=float(best.up_h), up_share=float(best.up_share)), t


FINAL = "--final" in sys.argv
NAMES = [a for a in sys.argv[1:] if a != "--final"]
PERIODS = ((2022, 2025),) if FINAL else ((2018, 2022),)
if FINAL:
    assert len(NAMES) == 1, "the sealed 2022 -> 2025 test is run for exactly one, final model"
    log = RES / "final_test_log.csv"
    prev = pd.read_csv(log) if log.exists() else pd.DataFrame(columns=["model", "time"])
    assert prev.empty, f"the sealed test was already run: {prev.to_dict('records')}"
    pd.DataFrame([{"model": NAMES[0], "time": pd.Timestamp.now().isoformat()}]).to_csv(log, index=False)
summary = []
for name in NAMES:
    pr = Predictor(name)
    rules_t, table = tune(pr)
    out = {"model": name, "tuned_rules": asdict(rules_t), "tuning": table.round(3).to_dict("records"), "tests": []}
    print(f"{name}: tuned pop-up rule up_h={rules_t.up_h} up_share={rules_t.up_share} "
          f"(val F1 {table.iloc[0].f1:.2f}, {int(table.iloc[0].val_popups)} LiDAR pop-ups in val)", flush=True)
    for z in M.TEST_ZIPS:
        for y0, y1 in PERIODS:
            pred = pr.raster(z, y0, y1)
            r = {"zip": z, "y0": y0, "y1": y1, **height_mae(pred, z, y1)}
            for lab, rules in (("tuned", rules_t), ("default", C.Rules())):
                r[lab] = compare(unit_flags(pred, z, y0, y1, rules), z, y0, y1)
            out["tests"].append(r)
            t_ = r["tuned"]
            print(f"  {z} {y0}->{y1}: bldg MAE {r['bldg_mae']:.2f} | new R {t_['new']['recall']:.2f} P "
                  f"{t_['new']['precision']:.2f} | pop-up R {t_['popup']['recall']:.2f} P {t_['popup']['precision']:.2f} "
                  f"(LiDAR {t_['popup']['lidar']}) | rise LiDAR/model {t_['popup_rise_median_lidar_vs_model'][0]:.1f}/"
                  f"{t_['popup_rise_median_lidar_vs_model'][1]:.1f} m", flush=True)
            for lab in ("tuned", "default"):
                q = r[lab]
                summary.append({"model": name, "zip": z, "period": f"{y0}-{y1}", "thresholds": lab,
                                "bldg_mae": r["bldg_mae"], "new_recall": q["new"]["recall"],
                                "new_precision": q["new"]["precision"], "popup_recall": q["popup"]["recall"],
                                "popup_precision": q["popup"]["precision"], "popup_f1": q["popup"]["f1"],
                                "lidar_popups": q["popup"]["lidar"]})
    M.save_json(out, RES / f"eval_{name}{'_FINAL' if FINAL else ''}.json")
sm = RES / ("final_summary.csv" if FINAL else "eval_summary.csv")
old = pd.read_csv(sm) if sm.exists() else pd.DataFrame()
if len(old):
    old = old[~old.model.isin(NAMES)]
pd.concat([old, pd.DataFrame(summary)]).to_csv(sm, index=False)
print("written", sm)
