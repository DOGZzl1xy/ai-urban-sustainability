"""Stage 3 v3: train / evaluate the unit-level change classifier (LiDAR-anchored, bi-temporal).

    python scripts/unitcls.py train [--zips 19132 19121 ...] [--iters 6000] [--tag v3]
    python scripts/unitcls.py dev  [--tag v3]       # development test: test ZIPs, 2018 -> 2022
    python scripts/unitcls.py final [--tag v3]      # SEALED test 2022 -> 2025: once, logged, refuses a second run

Training pairs: LiDAR 2018 -> 2022 in the training ZIPs. Unit labels: Stage 1 LiDAR flags (grew = new or pop-up;
popup). Validation: Stage 3 blocks (stratified by pop-ups); the checkpoint with the best validation pop-up average
precision is kept and the decision thresholds (max F1) are fixed on validation before any test run.
Outputs: data/interim/models/unitcls_{tag}.pt, results/bitemporal/unitcls_{tag}_{train.csv,val.json,dev.json,FINAL.json}
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bitemporal as B  # noqa: E402
import height_data as H  # noqa: E402
import height_model as M  # noqa: E402
import unitcls as U  # noqa: E402
from match import bootstrap_rate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["train", "dev", "final", "val"])
ap.add_argument("--zips", nargs="+", default=list(M.TRAIN_ZIPS))
ap.add_argument("--periods", nargs="+", default=["2018-2022"],
                help="training year pairs, e.g. 2015-2018 2010-2022; validation is always 2018-2022")
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--frozen-iters", type=int, default=3000)
ap.add_argument("--bs", type=int, default=6)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--tag", default="v3")
ap.add_argument("--random-init", action="store_true")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--loader", choices=["fast", "old"], default="fast",
                help="fast: memory-mapped .npy, DataLoader workers, GPU augmentation; old: single-process npz")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--rotate", action="store_true",
                help="add 90-degree rotations to the GPU augmentation (off: same augmentation as the old path)")
ap.add_argument("--log-every", type=int, default=0, help="per-step loss log every N steps (0 = off)")
ap.add_argument("--save-every", type=int, default=500, help="resumable checkpoint every N steps")
ap.add_argument("--resume", action="store_true", help="continue from data/interim/models/unitcls_{tag}_last.pt")
ap.add_argument("--dead-check", type=float, default=0.15,
                help="abort (exit 3) if validation grew-AP is below this at the first evaluation: dead unit head")
ap.add_argument("--guard-every-eval", action="store_true",
                help="apply the dead-check threshold at every evaluation, not only the first (M4 amendment 1)")
ap.add_argument("--pos-weight", type=float, default=5.0,
                help="sampling weight of pairs with >= 10 m2 of real change (fast loader); M4 fallback: 18.45")
ap.add_argument("--dead-check-at", type=int, default=0,
                help="step of the dead-head check (0 = the first regular evaluation); adds one evaluation there")
ap.add_argument("--pool", choices=["mean", "meanmax"], default="mean",
                help="unit pooling (M4 amendment 2 config S: meanmax)")
ap.add_argument("--ema", type=float, default=0.0,
                help="EMA decay of the weights, 0 = off (amendment 2 config S: 0.999, warm-up (1+t)/(10+t)); "
                     "validation, guard and saved checkpoints then use the EMA weights")
a = ap.parse_args()
RES = ROOT / ("results/citywide/m4" if H.CITY else "results/bitemporal")
CKPT = ROOT / f"data/interim/models/unitcls_{a.tag}.pt"          # best validation pop-up AP
CKPT_GREW = ROOT / f"data/interim/models/unitcls_{a.tag}_grew.pt"  # best validation grew AP
CKPT_LAST = ROOT / f"data/interim/models/unitcls_{a.tag}_last.pt"  # resumable state every --save-every steps
UNIT_OUT = ROOT / "data/interim/unitcls_preds"                    # per-unit predictions (unit ids, no addresses)
if __name__ == "__main__" and a.mode == "train" and (RES / "HOLD").exists():
    print(f"HOLD file present ({RES / 'HOLD'}): not starting training {a.tag}", flush=True)
    sys.exit(4)
CROP = 256


def prepare(zips, y0, y1):
    """Pair index with masks + unit-id paths, unit sizes and labels, keyed by (zip, y0, y1).

    Unit-id rasters are on the ZIP's LiDAR grid, which is the same for every year, so they are built once per ZIP
    from the 2018 patch index (a superset of the sparse-year patches)."""
    pi = B.build_masks(tuple(zips), y0, y1)
    sizes, labels = {}, {}
    for z in zips:
        for yr in sorted({y0, y1} - {2018}):  # unit-id rasters for patches that exist only in other years
            U.unit_id_patches(z, yr)
        sizes[(z, y0, y1)] = U.unit_id_patches(z, 2018)
        labels[(z, y0, y1)] = U.unit_labels(z, y0, y1)
    pi["uid"] = [str(U.UID_DIR / f"{z}_{i}_{j}.npy") for z, i, j in zip(pi.zip, pi.i, pi.j)]
    pi["key"] = [(z, y0, y1) for z in pi.zip]
    return pi, sizes, labels


def unit_targets(uniq, n, meta, keys_of_rows, sizes, labels):
    """For pooled units: (valid mask, grew, popup) using the crop's ZIP; units < 90% inside, uncovered or unknown
    are not scored; identity pairs (same image twice) are labelled 'no change'."""
    uniq, n = uniq.cpu().numpy(), n.cpu().numpy()
    samp, uid = uniq // 10_000_000, uniq % 10_000_000 - 1
    valid = np.zeros(len(uniq), bool)
    grew = np.zeros(len(uniq), np.float32)
    pop = np.zeros(len(uniq), np.float32)
    for q in range(len(uniq)):
        k, ident = int(meta[samp[q], 0]), bool(meta[samp[q], 3])
        z = keys_of_rows[k]
        tot = sizes[z].get(uid[q], 0)
        if tot == 0 or n[q] < U.MIN_IN_CROP * tot or uid[q] not in labels[z].index:
            continue
        lab = labels[z].loc[uid[q]]
        if not lab.covered:
            continue
        valid[q] = True
        if not ident:
            grew[q], pop[q] = lab.grew, lab.popup
    return valid, grew, pop


@torch.no_grad()
def score_units(model, pi, sizes, labels, shift):
    """Units fully inside a sliding window (stride 128) get the mean logit over those windows."""
    model.eval()
    acc, first = {}, {}
    offs = range(0, H.P - CROP + 1, 128)
    for z, i, j, p0, p1, up in zip(pi.key, pi.i, pi.j, pi.path0, pi.path1, pi.uid):
        a0, b0 = B.load_patch(p0, filled=False), B.load_patch(p1, filled=False)  # raw anchor, as in training data
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
                         "i": first[(z, u)][0], "j": first[(z, u)][1]})  # first patch scored (for block bootstrap)
    model.train()
    model.bb.eval()
    return pd.DataFrame(rows)


def ap_score(y, s):
    o = np.argsort(-s)
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def best_threshold(y, s):
    best = (0.0, 0.0)
    for t in np.unique(np.quantile(s, np.linspace(0.5, 0.999, 300))):
        p = s >= t
        tp = (p & (y == 1)).sum()
        f1 = 2 * tp / max(p.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (float(f1), float(t))
    return best


def report(df, thr):
    out = {"units": int(len(df))}
    for k in ("grew", "popup"):
        y, s = df[k].to_numpy(), df[f"logit_{k}"].to_numpy()
        pred = s >= thr[k]
        tp = int((pred & (y == 1)).sum())
        out[k] = {"lidar": int(y.sum()), "model": int(pred.sum()), "recall": tp / max(y.sum(), 1),
                  "precision": tp / max(pred.sum(), 1), "ap": ap_score(y, s),
                  "recall_ci": bootstrap_rate(pred[y == 1].astype(float)),
                  "precision_ci": bootstrap_rate(y[pred].astype(float))}
    return out


def make_loader(ds, workers, seed):
    if a.loader == "old":
        return torch.utils.data.DataLoader(ds, batch_size=a.bs)
    kw = dict(batch_size=a.bs, num_workers=workers, pin_memory=True,
              generator=torch.Generator().manual_seed(seed))
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4, worker_init_fn=B.fast_worker_init)
    return torch.utils.data.DataLoader(ds, **kw)


def load_pairs():
    """Training pairs (periods of --periods, train blocks) and validation pairs (2018 -> 2022, val blocks)."""
    if H.CITY:  # M4: citywide split (scripts/m4_patches.py, splits_citywide.md); same split for every year
        sp = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
        sp = sp[sp.split.isin(["train", "val"])][["zip", "i", "j", "split"]].drop_duplicates()
    else:
        sp = B.make_splits_bt(zips=tuple(a.zips))[["zip", "i", "j", "split"]]
    parts, sizes, labels = [], {}, {}
    for per in sorted(set(a.periods) | {"2018-2022"}):
        y0, y1 = (int(v) for v in per.split("-"))
        p_, s_, l_ = prepare(a.zips, y0, y1)
        p_ = p_.merge(sp, on=["zip", "i", "j"])  # same blocks for every period: no leakage across years
        p_["period"] = per
        parts.append(p_)
        sizes.update(s_)
        labels.update(l_)
    pi = pd.concat(parts, ignore_index=True)
    tr = pi[(pi.split == "train") & pi.period.isin(a.periods)].reset_index(drop=True)
    va = pi[(pi.split == "val") & (pi.period == "2018-2022")].reset_index(drop=True)
    return tr, va, sizes, labels


class EMA:
    """Exponential moving average of the trainable weights (decay min(d, (1+t)/(10+t)), so the early validations
    are not dominated by the initial weights). Kept in a full copy of the model, which is what is validated."""

    def __init__(self, model, decay):
        import copy

        self.model, self.decay, self.t = copy.deepcopy(model), decay, 0
        self.model.requires_grad_(False)
        self.track(model)

    def track(self, model):  # (re)build the list of tracked tensors, e.g. after unfreezing backbone blocks
        names = {n for n, p in model.named_parameters() if p.requires_grad}
        mine = dict(self.model.named_parameters())
        self.src = [p for n, p in model.named_parameters() if n in names]
        self.dst = [mine[n] for n, p in model.named_parameters() if n in names]

    @torch.no_grad()
    def update(self):
        self.t += 1
        d = min(self.decay, (1 + self.t) / (10 + self.t))
        torch._foreach_lerp_(self.dst, self.src, 1 - d)


def main_train():
    import psutil

    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tr, va, sizes, labels = load_pairs()
    print(f"unitcls {a.tag}: zips {a.zips}; periods {a.periods}; {len(tr)} train pairs "
          f"({tr.period.value_counts().to_dict()}) / {len(va)} val pairs (2018-2022); loader {a.loader}, "
          f"workers {a.workers}; bs {a.bs}, pos-weight {a.pos_weight}, pool {a.pool}, ema {a.ema}", flush=True)
    SHIFT = M.global_shift()
    model = U.UnitCls(random_init=a.random_init, pool=a.pool).cuda()
    if a.loader == "fast":
        model = model.to(memory_format=torch.channels_last)

    def optimizer(stage2=False):
        head = [p for n, p in model.named_parameters()
                if p.requires_grad and not n.startswith("bb.") and not n.startswith("unit_")]
        unit = [p for n, p in model.named_parameters() if n.startswith("unit_")]
        g = [{"params": head, "lr": 2e-4 if stage2 else 1e-3}, {"params": unit, "lr": 1e-4 if stage2 else 3e-4}]
        bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bb.")]
        return torch.optim.AdamW(g + ([{"params": bb, "lr": 2e-5}] if bb else []), weight_decay=1e-4)

    def scheduler(opt, stage2=False):
        n = a.iters - a.frozen_iters if stage2 else a.frozen_iters
        return torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=n,
                                                   pct_start=max(0.05, 2 / n))

    # ---- state (optionally resumed from the last periodic checkpoint)
    start, best_p, best_g, st = 0, -1.0, -1.0, {}
    opt = optimizer()
    sched = scheduler(opt)
    if a.resume and CKPT_LAST.exists():
        st = torch.load(CKPT_LAST, map_location="cpu", weights_only=False)
        start, best_p, best_g = st["iter"], st["best_popup"], st["best_grew"]
        if start >= a.frozen_iters:
            model.unfreeze_last(4)
            opt = optimizer(True)
            sched = scheduler(opt, True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        torch.cuda.set_rng_state(st["cuda_rng"])
        print(f"resumed from {CKPT_LAST.name} at iter {start} (data order after resume is re-seeded)", flush=True)
    ema = EMA(model, a.ema) if a.ema else None
    if ema and "ema" in st:
        ema.model.load_state_dict(st["ema"])
        ema.t = st["ema_t"]
    evm = ema.model if ema else model  # the weights that are validated, guarded and saved
    gen = torch.Generator(device="cuda").manual_seed(a.seed * 1000 + start)

    def new_loader(workers, offset):
        n = (a.iters - offset) * a.bs
        if a.loader == "old":
            ds = B.Pairs(tr.path0, tr.path1, n_per_epoch=n, shift=SHIFT, n_changed=tr.n_real,
                         masks=tr["mask"], uids=tr["uid"], seed=a.seed + offset)
        else:
            ds = B.FastPairs(tr.path0, tr.path1, tr["mask"], tr["uid"], tr.n_real, n, shift=SHIFT, seed=a.seed,
                             offset=offset, pos_weight=a.pos_weight)
        return make_loader(ds, workers, a.seed + offset)

    workers = a.workers
    loader = new_loader(workers, start)
    log_path = RES / f"unitcls_{a.tag}_train.csv"
    log = open(log_path, "a" if start else "w", newline="")
    w = csv.writer(log)
    if not start:
        w.writerow(["iter", "stage", "loss", "val_units", "val_popups", "val_ap_popup", "val_ap_grew", "sec",
                    "it_per_s", "workers"])
    steplog = open(RES / f"unitcls_{a.tag}_steps.csv", "a" if start else "w", newline="") if a.log_every else None
    sw = csv.writer(steplog) if steplog else None
    if sw and not start:
        sw.writerow(["iter", "loss", "sec", "avail_gb"])
    run, t0, t_last, it_last = [], time.time(), time.time(), start
    keys_of_rows = tr.key.tolist()
    model.train()
    model.bb.eval()
    it = start
    data = iter(loader)
    while it < a.iters:
        try:
            batch = next(data)
        except StopIteration:
            break
        it += 1
        if it == a.frozen_iters + 1 and start < it:
            model.unfreeze_last(4)
            opt = optimizer(True)
            sched = scheduler(opt, True)
            if ema:
                ema.track(model)
            print("stage 2: unfroze last 4 blocks", flush=True)
        if a.loader == "fast":
            batch = [t.cuda(non_blocking=True) if i < 10 else t for i, t in enumerate(batch)]
            x0, x1, h0, g0, m0, h1, g1, m1, rc, uc, meta = B.gpu_augment(batch, gen, rotate=a.rotate)
            x0 = x0.contiguous(memory_format=torch.channels_last)
            x1 = x1.contiguous(memory_format=torch.channels_last)
        else:
            x0, x1, h0, g0, m0, h1, g1, m1, rc, uc, meta = batch
            x0, x1, h0, g0, m0, h1, g1, m1, rc, uc = (t.cuda() for t in (x0, x1, h0, g0, m0, h1, g1, m1, rc, uc))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pix, (logits, uniq, n) = model(x0, x1, h0, g0, uc)
        loss = 0.5 * B.loss_fn("anchor", pix.float(), h0, g0, m0, h1, g1, m1, rc)
        valid, grew, pop = unit_targets(uniq, n, meta.numpy(), keys_of_rows, sizes, labels)
        if valid.any():
            vm = torch.from_numpy(valid).cuda()
            lg = logits.float()[vm]
            loss = loss + U.focal_bce(lg[:, 0], torch.from_numpy(grew[valid]).cuda()) \
                + U.focal_bce(lg[:, 1], torch.from_numpy(pop[valid]).cuda())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
        opt.step()
        sched.step()
        if ema:
            ema.update()
        run.append(loss.item())
        if sw and it % a.log_every == 0:
            sw.writerow([it, round(loss.item(), 4), round(time.time() - t0, 1),
                         round(psutil.virtual_memory().available / 1e9, 2)])
            steplog.flush()
        # memory guard: below 3 GB available, warn and fall back to 3 workers (slower, but no out-of-memory crash)
        if it % 50 == 0:
            avail = psutil.virtual_memory().available / 1e9
            if avail < 3 and a.loader == "fast" and workers > 3:
                print(f"MEMORY WARNING at iter {it}: {avail:.1f} GB available; workers {workers} -> 3", flush=True)
                workers = 3
                del data, loader
                loader = new_loader(workers, it)
                data = iter(loader)
            elif avail < 3:
                print(f"MEMORY WARNING at iter {it}: {avail:.1f} GB available", flush=True)
        check_at = a.dead_check_at or a.eval_every
        if it % a.eval_every == 0 or it == a.iters or it == check_at:
            speed = (it - it_last) / (time.time() - t_last)
            df = score_units(evm, va, sizes, labels, SHIFT)
            apg, app = ap_score(df.grew, df.logit_grew), ap_score(df.popup, df.logit_popup)
            w.writerow([it, 1 if it <= a.frozen_iters else 2, round(np.mean(run), 4), len(df), int(df.popup.sum()),
                        round(app, 4), round(apg, 4), round(time.time() - t0), round(speed, 3), workers])
            log.flush()
            print(f"{it:6d} loss {np.mean(run):.3f}  val units {len(df)} (pop-ups {int(df.popup.sum())})  "
                  f"AP pop-up {app:.3f}  AP grew {apg:.3f}  {time.time() - t0:.0f}s  {speed:.2f} it/s", flush=True)
            if (it == check_at or (a.guard_every_eval and it > check_at)) and apg < a.dead_check:
                print(f"DEAD RUN: grew AP {apg:.3f} < {a.dead_check} at step {it}; aborting", flush=True)
                sys.exit(3)
            run = []
            thr = {"grew": best_threshold(df.grew.to_numpy(), df.logit_grew.to_numpy())[1],
                   "popup": best_threshold(df.popup.to_numpy(), df.logit_popup.to_numpy())[1]}
            meta_ck = {"shift": SHIFT, "iter": it, "thresholds": thr, "zips": a.zips, "periods": a.periods,
                       "loader": a.loader, "pool": a.pool, "ema": a.ema}
            if app > best_p:  # each head keeps the checkpoint with its own best validation AP
                best_p = app
                torch.save({"state": evm.state_dict(), **meta_ck}, CKPT)
                M.save_json({"iter": it, "thresholds": thr, **report(df, thr)}, RES / f"unitcls_{a.tag}_val.json")
                UNIT_OUT.mkdir(parents=True, exist_ok=True)  # per-unit validation scores (amendment 2 bootstrap)
                df.assign(iter=it).to_parquet(UNIT_OUT / f"{a.tag}_valbest_2018_2022.parquet")
            if apg > best_g:
                best_g = apg
                torch.save({"state": evm.state_dict(), **meta_ck}, CKPT_GREW)
                M.save_json({"iter": it, "thresholds": thr, **report(df, thr)},
                            RES / f"unitcls_{a.tag}_val_grew.json")
            t_last, it_last = time.time(), it
        if it % a.save_every == 0 or it == a.iters:  # resumable state (model, optimizer, scheduler, RNG)
            tmp = CKPT_LAST.with_suffix(".tmp")
            torch.save({"iter": it, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(), "best_popup": best_p, "best_grew": best_g,
                        **({"ema": ema.model.state_dict(), "ema_t": ema.t} if ema else {})}, tmp)
            os.replace(tmp, CKPT_LAST)
    print(f"done {it} iters in {time.time() - t0:.0f}s; best val AP pop-up {best_p:.3f}, grew {best_g:.3f}")


if __name__ == "__main__" and a.mode == "train":
    main_train()
elif __name__ == "__main__" and a.mode == "val":
    # re-score the validation units (same pairs as in training) with the best pop-up checkpoint -> per-unit parquet
    # (used for checkpoints trained before training saved it, e.g. m4b_s0, for the amendment 2 bootstrap)
    _, va_, sizes_, labels_ = load_pairs()
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    mdl = U.UnitCls(random_init=True, pool=ck.get("pool", "mean"))
    mdl.load_state_dict(ck["state"])
    mdl = mdl.cuda()
    dfv = score_units(mdl, va_, sizes_, labels_, tuple(ck["shift"]))
    UNIT_OUT.mkdir(parents=True, exist_ok=True)
    dfv.assign(iter=ck["iter"]).to_parquet(UNIT_OUT / f"{a.tag}_valbest_2018_2022.parquet")
    print(f"{a.tag} (iter {ck['iter']}): {len(dfv)} val units, AP pop-up {ap_score(dfv.popup, dfv.logit_popup):.4f}, "
          f"AP grew {ap_score(dfv.grew, dfv.logit_grew):.4f}", flush=True)
elif __name__ == "__main__":
    final = a.mode == "final"
    y0, y1 = (2022, 2025) if final else (2018, 2022)
    if final:
        logp = RES / "final_test_log.csv"
        prev = pd.read_csv(logp) if logp.exists() else pd.DataFrame(columns=["model", "time"])
        assert prev.empty, f"the sealed test was already run: {prev.to_dict('records')}"
        pd.DataFrame([{"model": f"unitcls_{a.tag}", "time": pd.Timestamp.now().isoformat()}]).to_csv(logp,
                                                                                                    index=False)
    pi, sizes, labels = prepare(M.TEST_ZIPS, y0, y1)
    # validation units (training ZIPs, 2018 -> 2022, Stage 3 validation blocks): scored with the same checkpoints
    # and saved per unit, for count calibration (precision / recall) in the grid-level evaluation
    ck0 = torch.load(CKPT, map_location="cpu", weights_only=False)
    vzips = list(ck0.get("zips", M.TRAIN_ZIPS))
    pv, sv, lv = prepare(vzips, 2018, 2022)
    pv = pv.merge(B.make_splits_bt(zips=tuple(vzips))[["zip", "i", "j", "split"]], on=["zip", "i", "j"])
    pv = pv[pv.split == "val"].reset_index(drop=True)
    vals = {}

    def scored(path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = U.UnitCls(random_init=True, pool=ck.get("pool", "mean"))
        model.load_state_dict(ck["state"])
        model = model.cuda()
        if not final:
            vals[path] = score_units(model, pv, sv, lv, tuple(ck["shift"]))
        return ck, score_units(model, pi, sizes, labels, tuple(ck["shift"]))

    # each head is read from the checkpoint with its own best validation AP (grew-best falls back to the
    # pop-up-best checkpoint for runs trained before both were saved)
    ck_p, df = scored(CKPT)
    sources = {"popup": {"iter": ck_p["iter"], "checkpoint": CKPT.name}}
    thr = dict(ck_p["thresholds"])
    if CKPT_GREW.exists():
        ck_g, dg = scored(CKPT_GREW)
        df = df.drop(columns=["logit_grew"]).merge(dg[["zip", "unit", "logit_grew"]], on=["zip", "unit"])
        thr["grew"] = ck_g["thresholds"]["grew"]
        sources["grew"] = {"iter": ck_g["iter"], "checkpoint": CKPT_GREW.name}
    else:
        sources["grew"] = {"iter": ck_p["iter"], "checkpoint": CKPT.name, "note": "no grew-best checkpoint saved"}
    res = {"model": f"unitcls_{a.tag}", "period": f"{y0}-{y1}", "thresholds_from_validation": thr,
           "checkpoints": sources, "all": report(df, thr)}
    for z in M.TEST_ZIPS:
        res[z] = report(df[df.zip == z], thr)
    M.save_json(res, RES / f"unitcls_{a.tag}_{'FINAL' if final else 'dev'}.json")
    UNIT_OUT.mkdir(parents=True, exist_ok=True)
    df.assign(thr_grew=thr["grew"], thr_popup=thr["popup"]).to_parquet(
        UNIT_OUT / f"{a.tag}_{'FINAL' if final else 'dev'}_{y0}_{y1}.parquet")
    if vals:
        v = vals[CKPT]
        if CKPT_GREW in vals:
            v = v.drop(columns=["logit_grew"]).merge(vals[CKPT_GREW][["zip", "unit", "logit_grew"]],
                                                     on=["zip", "unit"])
        v.assign(thr_grew=thr["grew"], thr_popup=thr["popup"]).to_parquet(UNIT_OUT / f"{a.tag}_val_2018_2022.parquet")
    print(json.dumps(res["all"], indent=1, default=float))
