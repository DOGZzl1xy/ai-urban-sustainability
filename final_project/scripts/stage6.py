"""Stage 6 (results/citywide/stage6_plan.md): model scores for the intermediate years over the whole study area.

    POPUP_DATA=citywide python scripts/stage6.py select              # local: patch list -> data/interim/stage6_patches.csv
    POPUP_DATA=citywide python scripts/stage6.py infer [--workers 24] # cloud: score every unit, 4 pairs
    (aggregation: scripts/stage6_aggregate.py, local)

select: 512-cell patches of every ZIP window (46 ZIPs) with LiDAR height on >= 80% of cells in the anchor year
(2018 or 2022), >= 50% inside the ZIP and >= 25% inside the study-area mask (the round-2 rules).
infer: for each patch, anchor image + anchor nDSM / vegetation (as stored in training patches: nDSM float16, vegetation
> 0.5) and the target-year image (2019, 2020 for the 2018 anchor; 2023, 2024 for 2022), both warped exactly as in
scripts/m4_cut.py; pairs with >= 5% blank image in either year are skipped (pair_index rule). Units are scored as in
validation (256-cell windows at stride 128, unit >= 90% of its cells inside a window; mean logit over windows).
Cloud inputs: data/interim/anchor_rasters, anchor_restore (pasted back), anchor_units, data/raw/ortho/tiles,
unitcls_m4s_s0(.pt/_grew.pt); no web downloads (a missing tile is blank). Output, per ZIP and pair:
data/interim/stage6_scores/{ZIP}_{A}_{T}.parquet (zip, unit, anchor, target, n_cells, logit_grew, logit_popup).
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
from rasters import Grid  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
PAIRS = {2018: (2019, 2020), 2022: (2023, 2024)}
IDX = ROOT / "data/interim/stage6_patches.csv"
OUT = ROOT / "data/interim/stage6_scores"
ANCHOR_RAS = ROOT / "data/interim/anchor_rasters"
ANCHOR_UNITS = ROOT / "data/interim/anchor_units"
CROP = 256


def select():
    import geopandas as gpd
    from rasterio import features as rfeat
    from shapely.geometry import box

    import tiles as T
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    mask = gpd.read_file(ROOT / "data/interim/study_area_mask.gpkg", layer="mask").union_all()
    rows = []
    for z in sorted(zt.zip):
        zarea = T.study_area((z,), buffer_ft=0).union_all()
        for y in PAIRS:
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
                if valid < 0.8 or b.intersection(zarea).area / b.area < 0.5:
                    continue
                rows.append(dict(zip=z, anchor=y, i=p.i, j=p.j, x0=p.x0, y1=p.y1, valid=round(valid, 3)))
        print(f"  {z}: {sum(r['zip'] == z for r in rows)} patch-anchors", flush=True)
    idx = pd.DataFrame(rows)
    idx.to_csv(IDX, index=False)
    print(idx.groupby("anchor").size().to_string(), flush=True)


# ---------------------------------------------------------------- cloud inference
def _no_web():
    import ortho_chips as O
    O._fetch = lambda *a, **k: False   # no downloads on the cloud: a tile missing from the cache is blank


def _patch_inputs(args):
    """CPU worker: images of the anchor and both target years for one patch, warped as in scripts/m4_cut.py."""
    z, a, i, j, x0, y1 = args
    _no_web()
    p = H.Patch(int(i), int(j), float(x0), float(y1))
    imgs = {y: H.image_for(p, y, out_res_ft=H.CELL / 2, size=2 * H.P) for y in (a, *PAIRS[a])}
    blank = {y: float((im == 230).all(0).mean() + (im == 0).all(0).mean()) for y, im in imgs.items()}
    return z, a, i, j, imgs, blank


def infer(workers, zips=None):
    import geopandas as gpd
    import torch
    from rasterio import features as rfeat

    import change as C
    import height_model as M
    import unitcls as U

    torch.backends.cudnn.benchmark = True
    OUT.mkdir(parents=True, exist_ok=True)
    ckp = torch.load(ROOT / "data/interim/models/unitcls_m4s_s0.pt", map_location="cpu", weights_only=False)
    ckg = torch.load(ROOT / "data/interim/models/unitcls_m4s_s0_grew.pt", map_location="cpu", weights_only=False)
    same = all(torch.equal(ckp["state"][k], ckg["state"][k]) for k in ckp["state"])
    models = {}
    for name, ck in ([("both", ckp)] if same else [("popup", ckp), ("grew", ckg)]):
        m = U.UnitCls(random_init=True, pool=ck.get("pool", "mean"))
        m.load_state_dict(ck["state"])
        models[name] = m.cuda().eval()
    shift = tuple(ckp["shift"])
    print(f"M4 m4s_s0 (pop-up step {ckp['iter']}, grew step {ckg['iter']}; one model: {same}); shift {shift}",
          flush=True)
    idx = pd.read_csv(IDX, dtype={"zip": str})
    if zips:
        idx = idx[idx.zip.isin(zips)]
    # every batch-1 window that was blanked must have its restore patch, every ZIP its full raster (batch 2)
    need = pd.read_csv(ROOT / "data/interim/stage6_restore_needed.csv", dtype={"zip": str})
    miss = [f"{z}_{y}" for z, y in zip(need.zip, need.year) if z in set(idx.zip)
            and not (ROOT / f"data/interim/anchor_restore/{z}_{y}.npz").exists()]
    miss += [f"{z}_{a}" for z, a in idx[["zip", "anchor"]].drop_duplicates().itertuples(index=False)
             if not (ANCHOR_RAS / f"{z}_{a}.npz").exists()]
    assert not miss, f"batch 2 not in place: {miss[:6]} ..."
    t0 = time.time()
    for (z, a), sel in idx.groupby(["zip", "anchor"]):
        done = [OUT / f"{z}_{a}_{t}.parquet" for t in PAIRS[a]]
        if all(f.exists() for f in done):
            continue
        ras = np.load(ANCHOR_RAS / f"{z}_{a}.npz")
        nd, veg = ras["ndsm"].copy(), ras["veg"].copy()
        rf = ROOT / f"data/interim/anchor_restore/{z}_{a}.npz"
        if rf.exists():   # batch-1 blanked cells: paste the originals back
            r = np.load(rf)
            r0, c0 = int(r["r0"]), int(r["c0"])
            h, w = r["ndsm"].shape
            nd[r0:r0 + h, c0:c0 + w], veg[r0:r0 + h, c0:c0 + w] = r["ndsm"], r["veg"]
        grid = Grid(float(ras["x0"]), float(ras["y1"]), int(ras["nx"]), int(ras["ny"]))
        units = gpd.read_parquet(ANCHOR_UNITS / f"{z}.parquet")
        gi = units.geometry.buffer(-C.INSET_FT)
        ok = ~gi.is_empty
        uid = rfeat.rasterize(zip(gi[ok], units.unit[ok] + 1), out_shape=(grid.ny, grid.nx),
                              transform=grid.transform, fill=0, dtype="int32")
        sizes = U.unit_sizes(uid)
        acc = {t: {} for t in PAIRS[a]}
        jobs = [(z, a, r.i, r.j, r.x0, r.y1) for r in sel.itertuples()]
        with ProcessPoolExecutor(workers) as ex:
            for z_, a_, i, j, imgs, blank in ex.map(_patch_inputs, jobs, chunksize=1):
                sl = np.s_[i * H.P:(i + 1) * H.P, j * H.P:(j + 1) * H.P]
                h0 = nd[sl].astype(np.float16).astype(np.float32)          # as stored in the training patches
                g0 = veg[sl] > 0.5
                ud = uid[sl]
                im0 = M.shift_image(imgs[a], *shift)
                offs = range(0, H.P - CROP + 1, 128)
                pos = [(r_, c_) for r_ in offs for c_ in offs]
                x0 = torch.from_numpy(np.stack([im0[:, 2 * r_:2 * (r_ + CROP), 2 * c_:2 * (c_ + CROP)]
                                                for r_, c_ in pos]).astype(np.float32) / 255).cuda()
                hh = torch.from_numpy(np.stack([np.clip(np.nan_to_num(h0[r_:r_ + CROP, c_:c_ + CROP]), 0, M.H_MAX)
                                                for r_, c_ in pos])).cuda()
                gg = torch.from_numpy(np.stack([g0[r_:r_ + CROP, c_:c_ + CROP] for r_, c_ in pos])).cuda()
                uu = torch.from_numpy(np.stack([ud[r_:r_ + CROP, c_:c_ + CROP] for r_, c_ in pos])
                                      .astype(np.int64)).cuda()
                for t in PAIRS[a]:
                    if blank[a] >= 0.05 or blank[t] >= 0.05:
                        continue
                    im1 = M.shift_image(imgs[t], *shift)
                    x1 = torch.from_numpy(np.stack([im1[:, 2 * r_:2 * (r_ + CROP), 2 * c_:2 * (c_ + CROP)]
                                                    for r_, c_ in pos]).astype(np.float32) / 255).cuda()
                    out = {}
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        for name, m in models.items():
                            _, (lg, uniq, n) = m(x0, x1, hh, gg, uu)
                            out[name] = lg.float().cpu().numpy()
                    uniq, n = uniq.cpu().numpy(), n.cpu().numpy()
                    lgr = out["both"][:, 0] if same else out["grew"][:, 0]
                    lpo = out["both"][:, 1] if same else out["popup"][:, 1]
                    u_ = uniq % 10_000_000 - 1
                    for q in range(len(uniq)):
                        tot = sizes.get(u_[q], 0)
                        if tot and n[q] >= U.MIN_IN_CROP * tot:
                            acc[t].setdefault(int(u_[q]), []).append((lgr[q], lpo[q]))
        for t in PAIRS[a]:
            rows = [{"zip": z, "unit": u, "anchor": a, "target": t, "n_cells": int(sizes[u]),
                     "logit_grew": float(np.mean([v[0] for v in ls])), "logit_popup": float(np.mean([v[1] for v in ls]))}
                    for u, ls in acc[t].items()]
            pd.DataFrame(rows).to_parquet(OUT / f"{z}_{a}_{t}.parquet")
        print(f"  {z} {a}: {len(sel)} patches, units {[len(acc[t]) for t in PAIRS[a]]} "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("STAGE6 INFER DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "infer"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--zips", nargs="*")
    a = ap.parse_args()
    select() if a.step == "select" else infer(a.workers, a.zips)
