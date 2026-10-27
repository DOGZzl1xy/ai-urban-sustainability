"""Height-from-imagery models, data splits, dataset and loss for the pilot.

Models predict two maps on the 0.5 m label grid: height in metres and a vegetation logit (LiDAR multi-return
vegetation as the label), so the change rules can mask trees without LiDAR.
  unet          U-Net, ResNet-34 encoder with ImageNet weights (baseline B)
  dino_linear   frozen DINOv3 ViT-L/16 (SAT-493M) + linear head on the last layer (baseline A)
  dino_dpt      DINOv3 + DPT-style multi-level fusion head with a full-resolution image stem (main model)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import box
from shapely.ops import unary_union

import height_data as H
import tiles as T
from rasters import FT

ROOT = Path(__file__).resolve().parents[1]
DINO_ID = "facebook/dinov3-vitl16-pretrain-sat493m"
TRAIN_ZIPS = ("19132", "19121", "19143", "19125")
TEST_ZIPS = ("19146", "19147")
H_MAX = 60.0
SIDE_FT = H.P * H.CELL
SAT_MEAN, SAT_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)   # DINOv3 satellite normalisation
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- splits
def load_index(zips, year) -> pd.DataFrame:
    rows = []
    for z in zips:
        f = H.OUT / f"{z}_{year}" / "index.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"zip": str})
            d["path"] = [str(H.OUT / f"{z}_{year}" / f"{i}_{j}.npz") for i, j in zip(d.i, d.j)]
            rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["geometry"] = [box(x, y - SIDE_FT, x + SIDE_FT, y) for x, y in zip(d.x0, d.y1)]
    return d


def make_splits(year=2018, val_share=0.15, block_m=1000, buffer_m=50, test_buffer_m=100, seed=0) -> pd.DataFrame:
    """Train / val split of training-ZIP patches by 1 km blocks (on patch centres), with a 50 m buffer, and with
    every patch near a test ZIP removed. Overlapping patches from neighbouring ZIPs fall in the same block."""
    d = load_index(TRAIN_ZIPS, year)
    d = d[d.blank < 0.05].copy()
    test_area = T.study_area(TEST_ZIPS, buffer_ft=test_buffer_m / FT).union_all()
    d = d[[not g.intersects(test_area) for g in d.geometry]].copy()
    blk = block_m / FT
    c = [g.centroid for g in d.geometry]
    d["block"] = [f"{int(p.x // blk)}_{int(p.y // blk)}" for p in c]
    blocks = sorted(d.block.unique())
    rng = np.random.default_rng(seed)
    val_blocks = set(rng.choice(blocks, max(1, round(val_share * len(blocks))), replace=False))
    d["split"] = np.where(d.block.isin(val_blocks), "val", "train")
    vb = unary_union([g for g, s in zip(d.geometry, d.split) if s == "val"]).buffer(buffer_m / FT)
    drop = (d.split == "train") & np.array([g.intersects(vb) for g in d.geometry])
    d.loc[drop, "split"] = "buffer"
    return d.drop(columns="geometry")


# ---------------------------------------------------------------- registration
STAGE3_SHIFT = (-3, 0)   # global_shift() on the Stage 3 training ZIPs (0.75 m: imagery north of LiDAR)


def global_shift(year=2018, max_cells=4, min_corr=0.1):
    """Systematic image-vs-LiDAR offset, estimated on the TRAINING ZIPs only, in 0.25 m image pixels (dr, dc).

    The edge correlation finds image content ~0.8 m north of the LiDAR almost everywhere, consistent with the
    NAD83 / WGS84 datum difference that the default EPSG:2272 -> EPSG:3857 transform ignores. One constant
    shift is applied to every ZIP and year (also years without LiDAR). Per-patch outliers at the search edge are
    excluded from the estimate."""
    if H.CITY:  # M4 keeps the Stage 3 estimate (M4_plan.md: global image shift as in Stage 3)
        return STAGE3_SHIFT
    d = load_index(TRAIN_ZIPS, year)
    d = d[(d.off_r.abs() <= max_cells) & (d.off_c.abs() <= max_cells) & (d["corr"] > min_corr)]
    return int(round(2 * d.off_r.mean())), int(round(2 * d.off_c.mean()))


def shift_image(img, dr, dc):
    """Aligned image: out[r, c] = img[r + dr, c + dc] (edge pixels repeated)."""
    if dr == 0 and dc == 0:
        return img
    pad = max(abs(dr), abs(dc))
    p = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    h, w = img.shape[1:]
    return p[:, pad + dr:pad + dr + h, pad + dc:pad + dc + w]


# ---------------------------------------------------------------- data
class Patches(torch.utils.data.Dataset):
    """Random (train) or fixed tiled (eval) crops of `crop` label cells; image at 0.25 m or 0.5 m."""

    def __init__(self, paths, crop=256, img_res=0.25, train=True, n_per_epoch=None, seed=0, shift=(0, 0)):
        self.paths, self.crop, self.f, self.shift = list(paths), crop, 2 if img_res == 0.25 else 1, shift
        self.train, self.n = train, n_per_epoch or len(paths)
        self.cache = {}
        self.rng = np.random.default_rng(seed)
        if not train:  # deterministic 2x2 tiling of each 512 patch
            self.items = [(k, r, c) for k in range(len(self.paths)) for r in (0, H.P - crop) for c in (0, H.P - crop)]

    def __len__(self):
        return self.n if self.train else len(self.items)

    def _load(self, k):
        if k not in self.cache:
            d = np.load(self.paths[k])
            img = shift_image(d["img"], *self.shift)
            if self.f == 1:
                img = img.reshape(3, H.P, 2, H.P, 2).mean((2, 4)).astype(np.uint8)
            self.cache[k] = (img, d["ndsm"].astype(np.float32), d["veg"])
        return self.cache[k]

    def __getitem__(self, idx):
        if self.train:
            k = int(self.rng.integers(len(self.paths)))
            r, c = (int(v) for v in self.rng.integers(0, H.P - self.crop + 1, 2))
        else:
            k, r, c = self.items[idx]
        img, h, veg = self._load(k)
        f = self.f
        x = img[:, r * f:(r + self.crop) * f, c * f:(c + self.crop) * f].astype(np.float32) / 255
        y = h[r:r + self.crop, c:c + self.crop].copy()
        v = veg[r:r + self.crop, c:c + self.crop].copy()
        m = np.isfinite(y)                                   # LiDAR height available
        y = np.clip(np.nan_to_num(y), 0, H_MAX)
        if self.train:
            if self.rng.random() < 0.5:
                x, y, m, v = x[:, :, ::-1], y[:, ::-1], m[:, ::-1], v[:, ::-1]
            if self.rng.random() < 0.5:
                x, y, m, v = x[:, ::-1], y[::-1], m[::-1], v[::-1]
            # brightness / contrast / colour jitter: imagery differs between years
            x = (x - 0.5) * self.rng.uniform(0.8, 1.2) + 0.5 + self.rng.uniform(-0.08, 0.08)
            x = x * self.rng.uniform(0.92, 1.08, (3, 1, 1))
            x = np.clip(x, 0, 1).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(np.ascontiguousarray(m)), torch.from_numpy(np.ascontiguousarray(v)),
                torch.tensor([k, r, c]))


# ---------------------------------------------------------------- loss
def tolerant_l1(pred, y, m, tol=1):
    """Per-pixel min |pred - y| over label shifts within ±tol cells (absorbs residual misregistration)."""
    best = None
    yp = F.pad(y[:, None], (tol,) * 4, mode="replicate")[:, 0]
    mp = F.pad(m[:, None].float(), (tol,) * 4)[:, 0] > 0
    H_, W_ = y.shape[-2:]
    for dr in range(2 * tol + 1):
        for dc in range(2 * tol + 1):
            e = (pred - yp[:, dr:dr + H_, dc:dc + W_]).abs()
            e = torch.where(mp[:, dr:dr + H_, dc:dc + W_], e, torch.full_like(e, 1e4))
            best = e if best is None else torch.minimum(best, e)
    best = torch.where(m, best, torch.zeros_like(best))
    return best.sum() / m.sum().clamp(min=1)


def grad_l1(pred, y, m):
    lx = ((pred[..., 1:] - pred[..., :-1]) - (y[..., 1:] - y[..., :-1])).abs()
    ly = ((pred[..., 1:, :] - pred[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])).abs()
    mx, my = m[..., 1:] & m[..., :-1], m[..., 1:, :] & m[..., :-1, :]
    return ((lx * mx).sum() + (ly * my).sum()) / (mx.sum() + my.sum()).clamp(min=1)


def loss_fn(out, y, m, v, w_grad=0.5, w_veg=0.5):
    """out: (B, 2, H, W) = height, vegetation logit. Height is scored on LiDAR cells that are not vegetation."""
    h, vl = out[:, 0], out[:, 1]
    mh = m & ~v
    lv = F.binary_cross_entropy_with_logits(vl[m], v[m].float()) if m.any() else vl.sum() * 0
    return tolerant_l1(h, y, mh) + w_grad * grad_l1(h, y, mh) + w_veg * lv


def split_out(out):
    """(B, 2, H, W) raw output -> height (m), vegetation probability."""
    return out[:, 0], torch.sigmoid(out[:, 1])


# ---------------------------------------------------------------- models
def _norm(x, mean, std):
    return (x - x.new_tensor(mean)[:, None, None]) / x.new_tensor(std)[:, None, None]


class UNet(nn.Module):
    def __init__(self, img_res=0.25):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.net = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
        self.down = 2 if img_res == 0.25 else 1

    def forward(self, x):
        out = self.net(_norm(x, IMNET_MEAN, IMNET_STD))
        out = F.avg_pool2d(out, self.down) if self.down > 1 else out
        return torch.cat([F.softplus(out[:, :1]), out[:, 1:]], 1)


DINO_PTH = ROOT / "data/interim/models/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"  # Meta download, not in git


def convert_meta_state(sd: dict) -> dict:
    """Meta's original DINOv3 checkpoint -> transformers DINOv3ViTModel state dict.

    Fused qkv is split into q/k/v; the key bias is masked to zero in the original (bias_mask) and absent in HF.
    RoPE periods are not stored in HF: they follow rope_theta = 100, which matches the checkpoint's periods."""
    out = {"embeddings.cls_token": sd["cls_token"], "embeddings.mask_token": sd["mask_token"][None],
           "embeddings.register_tokens": sd["storage_tokens"],
           "embeddings.patch_embeddings.weight": sd["patch_embed.proj.weight"],
           "embeddings.patch_embeddings.bias": sd["patch_embed.proj.bias"],
           "norm.weight": sd["norm.weight"], "norm.bias": sd["norm.bias"]}
    n = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    for i in range(n):
        s, d = f"blocks.{i}.", f"model.layer.{i}."
        qkv_w, qkv_b = sd[s + "attn.qkv.weight"], sd[s + "attn.qkv.bias"] * sd[s + "attn.qkv.bias_mask"]
        dim = qkv_w.shape[1]
        q, k, v = qkv_w[:dim], qkv_w[dim:2 * dim], qkv_w[2 * dim:]
        out.update({d + "attention.q_proj.weight": q, d + "attention.k_proj.weight": k,
                    d + "attention.v_proj.weight": v, d + "attention.q_proj.bias": qkv_b[:dim],
                    d + "attention.v_proj.bias": qkv_b[2 * dim:],
                    d + "attention.o_proj.weight": sd[s + "attn.proj.weight"],
                    d + "attention.o_proj.bias": sd[s + "attn.proj.bias"],
                    d + "norm1.weight": sd[s + "norm1.weight"], d + "norm1.bias": sd[s + "norm1.bias"],
                    d + "norm2.weight": sd[s + "norm2.weight"], d + "norm2.bias": sd[s + "norm2.bias"],
                    d + "layer_scale1.lambda1": sd[s + "ls1.gamma"], d + "layer_scale2.lambda1": sd[s + "ls2.gamma"],
                    d + "mlp.up_proj.weight": sd[s + "mlp.fc1.weight"], d + "mlp.up_proj.bias": sd[s + "mlp.fc1.bias"],
                    d + "mlp.down_proj.weight": sd[s + "mlp.fc2.weight"],
                    d + "mlp.down_proj.bias": sd[s + "mlp.fc2.bias"]})
    return {k: v.float() for k, v in out.items()}


def load_dino(random_init=False):
    """ViT-L/16 satellite model (SAT-493M). Weights from Meta's download (DINO_PTH), else from Hugging Face."""
    from transformers import AutoModel, DINOv3ViTConfig, DINOv3ViTModel
    cfg = DINOv3ViTConfig(hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096,
                          num_register_tokens=4, patch_size=16, rope_theta=100.0, key_bias=False)
    if random_init:  # architecture only (testing, or a checkpoint that already holds the weights)
        return DINOv3ViTModel(cfg)
    if DINO_PTH.exists():
        m = DINOv3ViTModel(cfg)
        missing, unexpected = m.load_state_dict(
            convert_meta_state(torch.load(DINO_PTH, map_location="cpu", weights_only=True)), strict=False)
        assert not unexpected and all("inv_freq" in k for k in missing), (missing, unexpected)
        return m
    return AutoModel.from_pretrained(DINO_ID)


class _Fuse(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.res1 = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))
        self.res2 = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + skip + self.res1(skip)
        x = x + self.res2(x)
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


class Dino(nn.Module):
    """DINOv3 backbone + head. head='linear' (baseline A) or 'dpt' (main)."""

    def __init__(self, head="dpt", img_res=0.25, layers=(5, 11, 17, 23), feat=256, random_init=False):
        super().__init__()
        self.bb = load_dino(random_init)
        cfg = self.bb.config
        self.dim, self.nreg, self.ps = cfg.hidden_size, cfg.num_register_tokens, cfg.patch_size
        self.head_kind, self.layers, self.img_res = head, layers, img_res
        for p in self.bb.parameters():
            p.requires_grad = False
        if head == "linear":
            self.head = nn.Conv2d(self.dim, 2, 1)
        else:
            self.proj = nn.ModuleList([nn.Conv2d(self.dim, feat, 1) for _ in layers])
            # token grid t: layer outputs resized to 4t, 2t, t, t/2 (DPT reassemble)
            self.resize = nn.ModuleList([nn.ConvTranspose2d(feat, feat, 4, 4), nn.ConvTranspose2d(feat, feat, 2, 2),
                                         nn.Identity(), nn.Conv2d(feat, feat, 3, 2, 1)])
            self.fuse = nn.ModuleList([_Fuse(feat) for _ in layers])
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                      nn.ReLU())
            self.out = nn.Sequential(nn.Conv2d(feat + 32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    def unfreeze_last(self, k):
        for blk in self.bb.model.layer[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.bb.norm.parameters():
            p.requires_grad = True

    def tokens(self, x):
        o = self.bb(pixel_values=_norm(x, SAT_MEAN, SAT_STD), output_hidden_states=True)
        hs = o.hidden_states  # embeddings + each layer
        t = x.shape[-1] // self.ps

        def grid(h):
            return h[:, 1 + self.nreg:].transpose(1, 2).reshape(x.shape[0], self.dim, t, t)
        if self.head_kind == "linear":
            return [grid(self.bb.norm(o.last_hidden_state))]
        return [grid(hs[i + 1]) for i in self.layers]

    def forward(self, x):
        size = x.shape[-1] // (2 if self.img_res == 0.25 else 1)  # output on the 0.5 m label grid
        feats = self.tokens(x)
        if self.head_kind == "linear":
            out = F.interpolate(self.head(feats[0]), size=size, mode="bilinear", align_corners=False)
            return torch.cat([F.softplus(out[:, :1]), out[:, 1:]], 1)
        f = [rs(pr(z)) for z, pr, rs in zip(feats, self.proj, self.resize)]  # 4t, 2t, t, t/2
        y = self.fuse[3](f[3])
        y = self.fuse[2](y, f[2])
        y = self.fuse[1](y, f[1])
        y = self.fuse[0](y, f[0])                                           # 8t
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        s = self.stem(F.interpolate(x, size=size, mode="area"))
        out = self.out(torch.cat([y, s], 1))
        return torch.cat([F.softplus(out[:, :1]), out[:, 1:]], 1)


def build(name, img_res=0.25, random_init=False):
    if name == "unet":
        return UNet(img_res)
    return Dino("linear" if name == "dino_linear" else "dpt", img_res, random_init=random_init)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
