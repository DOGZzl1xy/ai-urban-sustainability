"""Train a height-from-imagery model on 2018 patches of the training ZIPs; select by validation-block MAE.

    python scripts/train_height.py unet|dino_linear|dino_dpt [--iters 6000] [--img-res 0.25] [--unfreeze 4]
                                   [--frozen-iters 3000] [--bs 8] [--tag NAME] [--random-init]

dino_dpt trains the head with a frozen backbone for --frozen-iters, then unfreezes the last --unfreeze blocks.
Writes data/interim/models/height_{tag}.pt, results/height_model_pilot/train_{tag}.csv and splits.csv.
The test ZIPs (19146, 19147) are never loaded here.
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
import height_model as M  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["unet", "dino_linear", "dino_dpt"])
ap.add_argument("--iters", type=int, default=6000)
ap.add_argument("--frozen-iters", type=int, default=3000)
ap.add_argument("--unfreeze", type=int, default=4)
ap.add_argument("--img-res", type=float, default=0.25)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--tag")
ap.add_argument("--random-init", action="store_true", help="DINOv3 architecture without pretrained weights (testing)")
a = ap.parse_args()
tag = a.tag or f"{a.model}_r{int(a.img_res * 100):03d}"
RES = ROOT / "results/height_model_pilot"
CKPT = ROOT / f"data/interim/models/height_{tag}.pt"
CKPT.parent.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)

sp = M.make_splits()
sp.to_csv(RES / "splits.csv", index=False)
tr, va = sp[sp.split == "train"].path.tolist(), sp[sp.split == "val"].path.tolist()
print(f"{tag}: {len(tr)} train / {len(va)} val patches ({(sp.split == 'buffer').sum()} dropped as buffer)", flush=True)
SHIFT = M.global_shift()
print(f"global image shift (0.25 m px, rows, cols): {SHIFT}", flush=True)
train = torch.utils.data.DataLoader(M.Patches(tr, img_res=a.img_res, n_per_epoch=a.iters * a.bs, shift=SHIFT),
                                    batch_size=a.bs)
val = torch.utils.data.DataLoader(M.Patches(va, img_res=a.img_res, train=False, shift=SHIFT), batch_size=a.bs)

model = M.build(a.model, a.img_res, random_init=a.random_init).cuda()


def optimizer(stage2=False):
    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("bb.")]
    groups = [{"params": head, "lr": 3e-4 if a.model == "unet" else (2e-4 if stage2 else 1e-3)}]
    bb = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("bb.")]
    if bb:
        groups.append({"params": bb, "lr": 2e-5})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


@torch.no_grad()
def evaluate():
    model.eval()
    ae = bae = n = bn = 0.0
    for x, y, m, v, _ in val:
        x, y, m, v = x.cuda(), y.cuda(), m.cuda(), v.cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x).float()[:, 0]
        m = m & ~v
        e = (p - y).abs()
        ae += (e * m).sum().item()
        n += m.sum().item()
        b = m & (y > 2.5)
        bae += (e * b).sum().item()
        bn += b.sum().item()
    model.train()
    if a.model.startswith("dino"):
        model.bb.eval()  # backbone stays in eval mode (no dropout / drop-path)
    return ae / n, bae / bn


opt = optimizer()
sched_len = a.frozen_iters if a.model == "dino_dpt" else a.iters
sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups], total_steps=sched_len,
                                            pct_start=max(0.05, 2 / sched_len))
log = open(RES / f"train_{tag}.csv", "w", newline="")
w = csv.writer(log)
w.writerow(["iter", "stage", "train_loss", "val_mae", "val_bldg_mae", "sec"])
best, run, t0 = np.inf, [], time.time()
model.train()
if a.model.startswith("dino"):
    model.bb.eval()
stage = 1
for it, (x, y, m, v, _) in enumerate(train, 1):
    if a.model == "dino_dpt" and it == a.frozen_iters + 1:
        stage = 2
        model.unfreeze_last(a.unfreeze)
        opt = optimizer(stage2=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, [g["lr"] for g in opt.param_groups],
                                                    total_steps=a.iters - a.frozen_iters,
                                                    pct_start=max(0.05, 2 / (a.iters - a.frozen_iters)))
        print(f"stage 2: unfroze last {a.unfreeze} blocks", flush=True)
    x, y, m, v = x.cuda(non_blocking=True), y.cuda(), m.cuda(), v.cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = model(x)
    loss = M.loss_fn(p.float(), y, m, v)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
    opt.step()
    sched.step()
    run.append(loss.item())
    if it % a.eval_every == 0 or it == a.iters:
        mae, bmae = evaluate()
        w.writerow([it, stage, round(np.mean(run), 4), round(mae, 4), round(bmae, 4), round(time.time() - t0)])
        log.flush()
        print(f"{it:6d} s{stage} loss {np.mean(run):.3f}  val MAE {mae:.3f} m  building-pixel MAE {bmae:.3f} m  "
              f"{time.time() - t0:.0f}s", flush=True)
        run = []
        if bmae < best:
            best = bmae
            torch.save({"model": a.model, "img_res": a.img_res, "iter": it, "state": model.state_dict(), "shift": SHIFT,
                        "val_bldg_mae": bmae, "random_init": a.random_init}, CKPT)
print(f"best val building-pixel MAE {best:.3f} m -> {CKPT.name}")
