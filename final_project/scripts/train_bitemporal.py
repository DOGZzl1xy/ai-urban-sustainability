"""Train a bi-temporal model on LiDAR 2018 -> 2022 changes in the training ZIPs (test ZIPs never loaded).

    python scripts/train_bitemporal.py pair|anchor [--iters 6000] [--frozen-iters 3000] [--bs 6] [--tag NAME]

Split: B.make_splits_bt (1 km blocks, stratified by ZIP and LiDAR pop-up count). Changed cells = real change
(B.change_mask_raster: inside LiDAR-flagged units). Checkpoint selection on the validation blocks: mean of
(height-change MAE on all cells, on real-change cells), so a model cannot win by predicting "no change" everywhere.
Writes data/interim/models/bitemp_{tag}.pt and results/bitemporal/train_{tag}.csv / .log.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bitemporal as B  # noqa: E402
import height_model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["pair", "anchor"])
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--frozen-iters", type=int, default=3000)
ap.add_argument("--unfreeze", type=int, default=4)
ap.add_argument("--bs", type=int, default=6)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--tag")
ap.add_argument("--random-init", action="store_true")
a = ap.parse_args()
tag = a.tag or a.variant
RES = ROOT / "results/bitemporal"
RES.mkdir(parents=True, exist_ok=True)
CKPT = ROOT / f"data/interim/models/bitemp_{tag}.pt"
torch.manual_seed(0)

pairs = B.build_masks(M.TRAIN_ZIPS, 2018, 2022)
sp = B.make_splits_bt()[["zip", "i", "j", "split"]]
pairs = pairs.merge(sp, on=["zip", "i", "j"])
pairs[["zip", "i", "j", "split", "n_real"]].to_csv(RES / "pairs_split.csv", index=False)
tr, va = pairs[pairs.split == "train"], pairs[pairs.split == "val"]
print(f"{tag}: {len(tr)} train / {len(va)} val pairs; real-change cells train {tr.n_real.sum():,} "
      f"(pairs with >= 10 m2: {(tr.n_real >= 40).sum()}), val {va.n_real.sum():,}", flush=True)
SHIFT = M.global_shift()
train = torch.utils.data.DataLoader(
    B.Pairs(tr.path0, tr.path1, n_per_epoch=a.iters * a.bs, shift=SHIFT, n_changed=tr.n_real,
            swap=a.variant == "pair", masks=tr["mask"]), batch_size=a.bs)
val = torch.utils.data.DataLoader(B.Pairs(va.path0, va.path1, train=False, shift=SHIFT, masks=va["mask"]),
                                  batch_size=a.bs)
model = B.BiTemporal(a.variant, random_init=a.random_init).cuda()


def optimizer(stage2=False):
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("bb.")]
    groups = [{"params": head, "lr": 2e-4 if stage2 else 1e-3}]
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bb.")]
    if bb:
        groups.append({"params": bb, "lr": 2e-5})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def predict_dh(out, h0):
    return out[:, 0] if a.variant == "pair" else out[:, 0] - h0


@torch.no_grad()
def evaluate():
    model.eval()
    s_all = n_all = s_ch = n_ch = 0.0
    for x0, x1, h0, g0, m0, h1, g1, m1, rc, _ in val:
        x0, x1, h0, g0, m0, h1, g1, m1, rc = (t.cuda() for t in (x0, x1, h0, g0, m0, h1, g1, m1, rc))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x0, x1, h0, g0).float()
        e = (predict_dh(out, h0) - (h1 - h0)).abs()
        both = m0 & m1 & ~g0 & ~g1
        ch = both & rc
        s_all += (e * both).sum().item()
        n_all += both.sum().item()
        s_ch += (e * ch).sum().item()
        n_ch += ch.sum().item()
    model.train()
    model.bb.eval()
    return s_all / n_all, s_ch / max(n_ch, 1)


opt = optimizer()
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=a.frozen_iters,
                                            pct_start=max(0.05, 2 / a.frozen_iters))
log = open(RES / f"train_{tag}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["iter", "stage", "train_loss", "val_dh_mae_all", "val_dh_mae_changed", "score", "sec"])
best, run, t0, stage = np.inf, [], time.time(), 1
model.train()
model.bb.eval()
for it, (x0, x1, h0, g0, m0, h1, g1, m1, rc, _) in enumerate(train, 1):
    if it == a.frozen_iters + 1:
        stage = 2
        model.unfreeze_last(a.unfreeze)
        opt = optimizer(stage2=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups],
                                                    total_steps=a.iters - a.frozen_iters,
                                                    pct_start=max(0.05, 2 / (a.iters - a.frozen_iters)))
        print(f"stage 2: unfroze last {a.unfreeze} blocks", flush=True)
    x0, x1, h0, g0, m0, h1, g1, m1, rc = (t.cuda(non_blocking=True) for t in (x0, x1, h0, g0, m0, h1, g1, m1, rc))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x0, x1, h0, g0)
    loss = B.loss_fn(a.variant, out.float(), h0, g0, m0, h1, g1, m1, rc)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
    opt.step()
    sched.step()
    run.append(loss.item())
    if it % a.eval_every == 0 or it == a.iters:
        mae_all, mae_ch = evaluate()
        score = (mae_all + mae_ch) / 2
        w.writerow([it, stage, round(np.mean(run), 4), round(mae_all, 4), round(mae_ch, 4), round(score, 4),
                    round(time.time() - t0)])
        log.flush()
        print(f"{it:6d} s{stage} loss {np.mean(run):.3f}  val dh MAE all {mae_all:.3f} m  changed {mae_ch:.3f} m  "
              f"{time.time() - t0:.0f}s", flush=True)
        run = []
        if score < best:
            best = score
            torch.save({"variant": a.variant, "iter": it, "state": model.state_dict(), "shift": SHIFT,
                        "val_dh_mae_all": mae_all, "val_dh_mae_changed": mae_ch}, CKPT)
print(f"best val score {best:.3f} m -> {CKPT.name}")
