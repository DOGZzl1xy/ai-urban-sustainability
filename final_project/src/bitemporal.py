"""Bi-temporal height models (Stage 3): detect vertical change from two aerial images.

Variant A ("pair"):    image t0 + image t1                 -> height change t1 - t0
Variant B ("anchor"):  LiDAR height t0 + image t0 + image t1 -> height at t1 (predicted as anchor + change)

Both share one DINOv3 ViT-L/16 (SAT-493M) encoder for the two images; per DPT level the two feature maps and their
difference are fused. Training pairs come from LiDAR 2018 -> 2022 in the training ZIPs only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import height_data as H
import height_model as M

ROOT = Path(__file__).resolve().parents[1]
CHANGE_M = 2.0          # |dh| above this counts as a changed pixel (weighting / sampling)


# ---------------------------------------------------------------- data
def pair_index(zips, y0, y1) -> pd.DataFrame:
    """Patches present in both years (same LiDAR grid, so same i, j)."""
    a, b = M.load_index(zips, y0), M.load_index(zips, y1)
    k = ["zip", "i", "j"]
    d = a[k + ["x0", "y1", "path", "blank"]].merge(b[k + ["path", "blank"]], on=k, suffixes=("0", "1"))
    return d[(d.blank0 < 0.05) & (d.blank1 < 0.05)].reset_index(drop=True)


def change_stats(paths0, paths1):
    """Changed-cell count per patch pair (LiDAR, both non-vegetation)."""
    out = []
    for p0, p1 in zip(paths0, paths1):
        a, b = np.load(p0), np.load(p1)
        h0, h1 = a["ndsm"].astype(np.float32), b["ndsm"].astype(np.float32)
        ok = np.isfinite(h0) & np.isfinite(h1) & ~a["veg"] & ~b["veg"]
        out.append(int((ok & (np.abs(h1 - h0) > CHANGE_M) & (np.maximum(h0, h1) > 2.5)).sum()))
    return np.array(out)


class Pairs(torch.utils.data.Dataset):
    """Crops of (img0, img1, h0, veg0, h1, veg1, valid0, valid1) on the 0.5 m grid; images at 0.25 m."""

    def __init__(self, paths0, paths1, crop=256, train=True, n_per_epoch=None, seed=0, shift=(0, 0),
                 n_changed=None, swap=False, p_identity=0.2, masks=None, uids=None):
        self.p0, self.p1, self.crop, self.train, self.shift = list(paths0), list(paths1), crop, train, shift
        self.masks = list(masks) if masks is not None else None
        self.uids = list(uids) if uids is not None else None  # unit-id rasters (unitcls); returned when given
        self.swap, self.p_identity = swap, p_identity
        self.rng = np.random.default_rng(seed)
        self.cache = {}
        if train:
            self.n = n_per_epoch
            w = np.ones(len(self.p0)) if n_changed is None else 1 + 4 * (np.asarray(n_changed) >= 40)  # >= 10 m2
            self.w = w / w.sum()
        else:
            self.items = [(k, r, c) for k in range(len(self.p0)) for r in (0, H.P - crop) for c in (0, H.P - crop)]

    def __len__(self):
        return self.n if self.train else len(self.items)

    def _load(self, k):
        if k not in self.cache:
            a, b = np.load(self.p0[k]), np.load(self.p1[k])
            mk = np.load(self.masks[k]) if self.masks is not None else np.zeros((H.P, H.P), bool)
            ud = np.load(self.uids[k]) if self.uids is not None else np.zeros((H.P, H.P), np.int32)
            h0 = a["ndsm"].astype(np.float32)
            if np.isfinite(h0).mean() < 0.8:  # sparse anchor year (2010 / 2015): nearest-neighbour fill for the input
                from scipy import ndimage
                idx = ndimage.distance_transform_edt(~np.isfinite(h0), return_distances=False, return_indices=True)
                h0 = h0[tuple(idx)]
            self.cache[k] = (M.shift_image(a["img"], *self.shift), h0, a["veg"],
                             M.shift_image(b["img"], *self.shift), b["ndsm"].astype(np.float32), b["veg"], mk, ud)
        return self.cache[k]

    def _jitter(self, x, strong=False):
        s = 2.0 if strong else 1.0
        x = (x - 0.5) * self.rng.uniform(1 - 0.2 * s, 1 + 0.2 * s) + 0.5 + self.rng.uniform(-0.08 * s, 0.08 * s)
        x = x * self.rng.uniform(1 - 0.08 * s, 1 + 0.08 * s, (3, 1, 1))
        if strong and self.rng.random() < 0.5:  # different sun / blur
            x = x * self.rng.uniform(0.6, 1.0)
        return np.clip(x, 0, 1)

    def __getitem__(self, idx):
        if self.train:
            k = int(self.rng.choice(len(self.p0), p=self.w))
        else:
            k, r, c = self.items[idx]
        i0, h0, v0, i1, h1, v1, mk, ud = self._load(k)
        if self.train:
            ch = mk  # real change only (see change_mask_raster)
            if ch.any() and self.rng.random() < 0.7:  # centre the crop near a changed cell
                rr, cc = np.nonzero(ch)
                q = int(self.rng.integers(len(rr)))
                r = int(np.clip(rr[q] - self.crop // 2 + self.rng.integers(-64, 65), 0, H.P - self.crop))
                c = int(np.clip(cc[q] - self.crop // 2 + self.rng.integers(-64, 65), 0, H.P - self.crop))
            else:
                r, c = (int(v) for v in self.rng.integers(0, H.P - self.crop + 1, 2))
        sl = np.s_[r:r + self.crop, c:c + self.crop]
        sli = np.s_[:, 2 * r:2 * (r + self.crop), 2 * c:2 * (c + self.crop)]
        x0, x1 = i0[sli].astype(np.float32) / 255, i1[sli].astype(np.float32) / 255
        a0, a1, g0, g1 = h0[sl].copy(), h1[sl].copy(), v0[sl].copy(), v1[sl].copy()
        rc = mk[sl].copy()
        uc = ud[sl].copy()
        identity = False
        if self.train:
            if self.rng.random() < self.p_identity:  # same scene, different look -> no change
                x1, a1, g1, identity = x0.copy(), a0.copy(), g0.copy(), True
                rc = np.zeros_like(rc)
            if self.swap and self.rng.random() < 0.5:
                x0, x1, a0, a1, g0, g1 = x1, x0, a1, a0, g1, g0
            flips = [self.rng.random() < 0.5, self.rng.random() < 0.5]
            arrs = [x0, x1, a0, a1, g0, g1, rc, uc]
            if flips[0]:
                arrs = [z[..., ::-1] for z in arrs]
            if flips[1]:
                arrs = [z[..., ::-1, :] for z in arrs]
            x0, x1, a0, a1, g0, g1, rc, uc = arrs
            x0, x1 = self._jitter(x0), self._jitter(x1, strong=identity)
        m0, m1 = np.isfinite(a0), np.isfinite(a1)
        a0, a1 = np.clip(np.nan_to_num(a0), 0, M.H_MAX), np.clip(np.nan_to_num(a1), 0, M.H_MAX)
        t = lambda z: torch.from_numpy(np.ascontiguousarray(z))  # noqa: E731
        meta = torch.tensor([k, r, c, int(identity)])
        if self.uids is not None:
            return (t(x0.astype(np.float32)), t(x1.astype(np.float32)), t(a0), t(g0), t(m0), t(a1), t(g1), t(m1),
                    t(rc), t(uc.astype(np.int64)), meta)
        return (t(x0.astype(np.float32)), t(x1.astype(np.float32)), t(a0), t(g0), t(m0), t(a1), t(g1), t(m1),
                t(rc), meta)


# ---------------------------------------------------------------- loss
def weighted_tolerant_l1(pred, y, m, w, tol=1):
    best = None
    yp = F.pad(y[:, None], (tol,) * 4, mode="replicate")[:, 0]
    mp = F.pad(m[:, None].float(), (tol,) * 4)[:, 0] > 0
    h, wd = y.shape[-2:]
    for dr in range(2 * tol + 1):
        for dc in range(2 * tol + 1):
            e = (pred - yp[:, dr:dr + h, dc:dc + wd]).abs()
            e = torch.where(mp[:, dr:dr + h, dc:dc + wd], e, torch.full_like(e, 1e4))
            best = e if best is None else torch.minimum(best, e)
    best = torch.where(m, best, torch.zeros_like(best))
    return (best * w).sum() / (w * m).sum().clamp(min=1)


def loss_fn(variant, out, h0, g0, m0, h1, g1, m1, real, w_change=4.0, w_veg=0.5):
    """out: (B, 2, H, W). Pair: channel 0 = dh. Anchor: channel 0 = h1 (already anchor + dh). Channel 1 = veg logit.
    real: real-change cells (change_mask_raster), up-weighted."""
    both = m0 & m1 & ~g0 & ~g1
    dh = h1 - h0
    changed = both & real
    w = 1 + w_change * changed.float()
    if variant == "pair":
        loss = weighted_tolerant_l1(out[:, 0], dh, both, w) + 0.5 * M.grad_l1(out[:, 0], dh, both)
    else:
        mh = m1 & ~g1
        loss = weighted_tolerant_l1(out[:, 0], h1, mh, w) + 0.5 * M.grad_l1(out[:, 0], h1, mh)
    lv = F.binary_cross_entropy_with_logits(out[:, 1][m1], g1[m1].float()) if m1.any() else out.sum() * 0
    return loss + w_veg * lv


# ---------------------------------------------------------------- model
class BiTemporal(nn.Module):
    def __init__(self, variant="anchor", layers=(5, 11, 17, 23), feat=256, random_init=False):
        super().__init__()
        assert variant in ("pair", "anchor")
        self.variant, self.layers = variant, layers
        self.bb = M.load_dino(random_init)
        cfg = self.bb.config
        self.dim, self.nreg, self.ps = cfg.hidden_size, cfg.num_register_tokens, cfg.patch_size
        for p in self.bb.parameters():
            p.requires_grad = False
        self.proj = nn.ModuleList([nn.Conv2d(3 * self.dim, feat, 1) for _ in layers])  # [f0, f1, f1 - f0]
        self.resize = nn.ModuleList([nn.ConvTranspose2d(feat, feat, 4, 4), nn.ConvTranspose2d(feat, feat, 2, 2),
                                     nn.Identity(), nn.Conv2d(feat, feat, 3, 2, 1)])
        self.fuse = nn.ModuleList([M._Fuse(feat) for _ in layers])
        self.stem = nn.Sequential(nn.Conv2d(6, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        extra = 0
        if variant == "anchor":  # anchor height + vegetation at full output resolution
            self.anchor = nn.Sequential(nn.Conv2d(2, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                        nn.ReLU())
            self.anchor_tok = nn.Conv2d(2, feat, 1)  # anchor injected at token resolution too
            extra = 32
        self.out = nn.Sequential(nn.Conv2d(feat + 32 + extra, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 2, 1))

    unfreeze_last = M.Dino.unfreeze_last

    def _tokens(self, x):
        o = self.bb(pixel_values=M._norm(x, M.SAT_MEAN, M.SAT_STD), output_hidden_states=True)
        t = x.shape[-1] // self.ps
        return [o.hidden_states[i + 1][:, 1 + self.nreg:].transpose(1, 2).reshape(x.shape[0], self.dim, t, t)
                for i in self.layers]

    def forward(self, x0, x1, h0=None, g0=None):
        b = x0.shape[0]
        size = x0.shape[-1] // 2                      # 0.25 m images -> 0.5 m output
        feats = self._tokens(torch.cat([x0, x1]))
        f = []
        for z, pr, rs in zip(feats, self.proj, self.resize):
            a, c = z[:b], z[b:]
            f.append(rs(pr(torch.cat([a, c, c - a], 1))))
        if self.variant == "anchor":
            an = torch.stack([h0 / 10, g0.float()], 1)
            f[2] = f[2] + self.anchor_tok(F.adaptive_avg_pool2d(an, f[2].shape[-1]))
        y = self.fuse[3](f[3])
        y = self.fuse[2](y, f[2])
        y = self.fuse[1](y, f[1])
        y = self.fuse[0](y, f[0])
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        s = self.stem(F.interpolate(torch.cat([x0, x1], 1), size=size, mode="area"))
        z = [y, s] + ([self.anchor(an)] if self.variant == "anchor" else [])
        out = self.out(torch.cat(z, 1))
        if self.variant == "anchor":  # predict the change and add it to the known height
            return torch.cat([F.relu(h0[:, None] + out[:, :1]), out[:, 1:]], 1)
        return out


# ---------------------------------------------------------------- real-change masks and Stage 3 split
MASK_DIR = ROOT / ("data/interim/bitemporal_masks_city" if H.CITY else "data/interim/bitemporal_masks")


def change_mask_raster(zip_code, y0, y1):
    """Cells of real change: inside a unit that the Stage 1 LiDAR rules flag (new, pop-up or demolition; precision
    95.5% on a fresh blind sample), |dh| > 2.5 m and non-vegetation in both years. Raw |dh| alone is dominated by
    roof-edge points, vehicles and leaf-off trees."""
    import geopandas as gpd
    from rasterio import features as rfeat
    from rasters import Grid

    a, b = np.load(H.raster_path(zip_code, y0)), np.load(H.raster_path(zip_code, y1))
    grid = Grid(float(a["x0"]), float(a["y1"]), int(a["nx"]), int(a["ny"]))
    f = pd.read_parquet(H.change_path(zip_code, y0, y1))
    flagged = set(f[f.chg_new | f.chg_popup | f.chg_demo].unit)
    units = gpd.read_parquet(H.units_path(zip_code))
    units = units[units.unit.isin(flagged)]
    inside = rfeat.rasterize(((g, 1) for g in units.geometry), out_shape=(grid.ny, grid.nx),
                             transform=grid.transform, fill=0, dtype="uint8").astype(bool)
    dh = b["ndsm"] - a["ndsm"]
    ok = np.isfinite(dh) & ~(np.nan_to_num(a["veg"]) > 0.5) & ~(np.nan_to_num(b["veg"]) > 0.5)
    return inside & ok & (np.abs(np.nan_to_num(dh)) > 2.5)


def build_masks(zips, y0, y1):
    """Per-patch real-change masks (bool 512 x 512) for every pair; returns pair index with n_real."""
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    pi = pair_index(zips, y0, y1)
    n = []
    import unitcls as U  # lazy import: unitcls imports this module
    for z in ([] if U.PREP_CACHE else zips):  # cache: masks were built before and are read as they are
        mk = change_mask_raster(z, y0, y1)
        for i, j in zip(pi[pi.zip == z].i, pi[pi.zip == z].j):
            m = mk[i * H.P:(i + 1) * H.P, j * H.P:(j + 1) * H.P]
            np.save(MASK_DIR / f"{z}_{y0}_{y1}_{i}_{j}.npy", m)
    for z, i, j in zip(pi.zip, pi.i, pi.j):
        n.append(int(np.load(MASK_DIR / f"{z}_{y0}_{y1}_{i}_{j}.npy").sum()))
    pi["n_real"] = n
    pi["mask"] = [str(MASK_DIR / f"{z}_{y0}_{y1}_{i}_{j}.npy") for z, i, j in zip(pi.zip, pi.i, pi.j)]
    return pi


def make_splits_bt(y0=2018, y1=2022, val_share=0.15, block_m=1000, buffer_m=50, test_buffer_m=100,
                   zips=M.TRAIN_ZIPS):
    """Stage 3 split of training-ZIP patches, stratified by ZIP and by LiDAR pop-up count.

    Within each ZIP, 1 km blocks are ordered by the number of LiDAR-flagged pop-ups (unit centroids) and every
    k-th block goes to validation (k ≈ 1 / val_share), so validation sees a representative amount of change.
    Uses only fixed rules and LiDAR labels, no model output. Same 50 m buffer and test-ZIP exclusion as Stage 2."""
    import geopandas as gpd
    from shapely.ops import unary_union

    import tiles as T
    from rasters import FT

    d = M.load_index(zips, y0)
    d = d[d.blank < 0.05].copy()
    test_area = T.study_area(M.TEST_ZIPS, buffer_ft=test_buffer_m / FT).union_all()
    d = d[[not g.intersects(test_area) for g in d.geometry]].copy()
    blk = block_m / FT
    d["block"] = [f"{int(g.centroid.x // blk)}_{int(g.centroid.y // blk)}" for g in d.geometry]
    val_blocks = set()
    for z in zips:
        f = pd.read_parquet(ROOT / f"data/interim/lidar_change/{z}_{y0}_{y1}.parquet")
        u = gpd.read_parquet(ROOT / f"data/interim/units_{z}.parquet")[["unit", "geometry"]].merge(
            f[["unit", "chg_popup"]], on="unit")
        c = u[u.chg_popup].geometry.centroid
        pops = pd.Series([f"{int(p.x // blk)}_{int(p.y // blk)}" for p in c]).value_counts()
        blocks = sorted(d[d.zip == z].block.unique(), key=lambda b: (-pops.get(b, 0), b))
        k = max(2, round(1 / val_share))
        val_blocks |= set(blocks[k // 2::k])
    d["split"] = np.where(d.block.isin(val_blocks), "val", "train")
    vb = unary_union([g for g, s in zip(d.geometry, d.split) if s == "val"]).buffer(buffer_m / FT)
    drop = (d.split == "train") & np.array([g.intersects(vb) for g in d.geometry])
    d.loc[drop, "split"] = "buffer"
    return d.drop(columns="geometry")


# ---------------------------------------------------------------- fast data path (memory-mapped .npy, GPU augmentation)
NPY = ROOT / ("data/interim/height_npy_city" if H.CITY else "data/interim/height_npy")


def npy_stem(npz_path):
    """data/interim/height/{zip}_{year}/{i}_{j}.npz -> data/interim/height_npy/{zip}_{year}/{i}_{j}"""
    p = Path(npz_path)
    return NPY / p.parent.name / p.stem


def load_patch(npz_path, mmap=True, filled=True):
    """img / ndsm / veg of one patch; the unpacked .npy files (scripts/unpack_patches.py) when present."""
    s = npy_stem(npz_path)
    if (s.parent / f"{s.name}_veg.npy").exists():
        mode = "r" if mmap else None
        f = s.parent / f"{s.name}_ndsm_filled.npy"
        nd = np.load(f if filled and f.exists() else s.parent / f"{s.name}_ndsm.npy", mmap_mode=mode)
        return {"img": np.load(s.parent / f"{s.name}_img.npy", mmap_mode=mode), "ndsm": nd,
                "veg": np.load(s.parent / f"{s.name}_veg.npy", mmap_mode=mode)}
    d = np.load(npz_path)
    return {"img": d["img"], "ndsm": d["ndsm"].astype(np.float32), "veg": d["veg"]}


def crop_shifted(img, y0, x0, size, dr, dc):
    """img[:, y0+dr : y0+dr+size, x0+dc : x0+dc+size] with edge pixels repeated outside the array (uint8 copy).
    Same result as cropping M.shift_image(img, dr, dc) at (y0, x0), without shifting the whole patch."""
    h, w = img.shape[1:]
    ys = np.clip(np.arange(y0 + dr, y0 + dr + size), 0, h - 1)
    xs = np.clip(np.arange(x0 + dc, x0 + dc + size), 0, w - 1)
    if ys[0] == y0 + dr and ys[-1] == y0 + dr + size - 1 and xs[0] == x0 + dc and xs[-1] == x0 + dc + size - 1:
        return np.array(img[:, ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1])
    return np.array(img[:, ys][:, :, xs])


class FastPairs(torch.utils.data.Dataset):
    """Pairs for the unit classifier, read from memory-mapped .npy. Returns uint8 images (augmented on the GPU by
    gpu_augment); crop choice, identity pairs and label crops as in Pairs. Each DataLoader worker gets its own
    numpy RNG from worker_init (seed, worker id, offset), so a run is reproducible for a fixed number of workers."""

    def __init__(self, paths0, paths1, masks, uids, n_changed, n_per_epoch, crop=256, shift=(0, 0), seed=0,
                 p_identity=0.2, offset=0, pos_weight=5.0):
        self.p0, self.p1, self.masks, self.uids = list(paths0), list(paths1), list(masks), list(uids)
        w = 1 + (pos_weight - 1) * (np.asarray(n_changed) >= 40)  # pairs with >= 10 m2 of real change
        self.w = w / w.sum()
        self.n, self.crop, self.shift, self.seed, self.p_identity = n_per_epoch, crop, shift, seed, p_identity
        self.offset = offset
        self.rng = np.random.default_rng([seed, 99, offset])  # main-process fallback (num_workers = 0)
        self._mm = {}

    def __len__(self):
        return self.n

    def _get(self, k):
        if k not in self._mm:
            a, b = load_patch(self.p0[k]), load_patch(self.p1[k])
            self._mm[k] = (a, b, np.load(self.masks[k], mmap_mode="r"), np.load(self.uids[k], mmap_mode="r"))
        return self._mm[k]

    def __getitem__(self, idx):
        rng = self.rng
        k = int(rng.choice(len(self.p0), p=self.w))
        a, b, mk, ud = self._get(k)
        C = self.crop
        if rng.random() < 0.7:
            rr, cc = np.nonzero(mk)
            if len(rr):
                q = int(rng.integers(len(rr)))
                r = int(np.clip(rr[q] - C // 2 + rng.integers(-64, 65), 0, H.P - C))
                c = int(np.clip(cc[q] - C // 2 + rng.integers(-64, 65), 0, H.P - C))
            else:
                r, c = (int(v) for v in rng.integers(0, H.P - C + 1, 2))
        else:
            r, c = (int(v) for v in rng.integers(0, H.P - C + 1, 2))
        sl = np.s_[r:r + C, c:c + C]
        dr, dc = self.shift
        x0 = crop_shifted(a["img"], 2 * r, 2 * c, 2 * C, dr, dc)
        h0, g0 = np.array(a["ndsm"][sl], np.float32), np.array(a["veg"][sl])
        identity = rng.random() < self.p_identity
        if identity:  # same scene, different look -> no change
            x1, h1, g1, rc = x0.copy(), h0.copy(), g0.copy(), np.zeros((C, C), bool)
        else:
            x1 = crop_shifted(b["img"], 2 * r, 2 * c, 2 * C, dr, dc)
            h1, g1, rc = np.array(b["ndsm"][sl], np.float32), np.array(b["veg"][sl]), np.array(mk[sl])
        m0, m1 = np.isfinite(h0), np.isfinite(h1)
        h0, h1 = np.clip(np.nan_to_num(h0), 0, M.H_MAX), np.clip(np.nan_to_num(h1), 0, M.H_MAX)
        t = torch.from_numpy
        return (t(x0), t(x1), t(h0), t(g0), t(m0), t(h1), t(g1), t(m1), t(rc), t(np.array(ud[sl], np.int32)),
                torch.tensor([k, r, c, int(identity)]))


def fast_worker_init(worker_id):
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng([ds.seed, worker_id, ds.offset])


@torch.no_grad()
def gpu_augment(batch, gen, rotate=False):
    """On the GPU: uint8 -> float, per-sample flips and optional 90-degree rotations (same for every map of a sample),
    brightness / contrast / colour jitter per image (strong for identity pairs). batch as returned by FastPairs."""
    x0, x1, h0, g0, m0, h1, g1, m1, rc, uc, meta = batch
    dev = x0.device
    bsz = x0.shape[0]
    x0, x1 = x0.float() / 255, x1.float() / 255
    uc = uc.long()
    ident = meta[:, 3].to(dev).bool()

    def jit(x, strong):
        s = torch.where(strong, 2.0, 1.0)[:, None, None, None]
        u = lambda lo, hi, shape: lo + (hi - lo) * torch.rand(shape, device=dev, generator=gen)  # noqa: E731
        con = 1 + (u(-0.2, 0.2, (bsz, 1, 1, 1)) * s)
        bri = u(-0.08, 0.08, (bsz, 1, 1, 1)) * s
        col = 1 + u(-0.08, 0.08, (bsz, 3, 1, 1)) * s
        x = ((x - 0.5) * con + 0.5 + bri) * col
        dim = torch.where(strong[:, None, None, None] & (torch.rand((bsz, 1, 1, 1), device=dev, generator=gen) < 0.5),
                          u(0.6, 1.0, (bsz, 1, 1, 1)), torch.ones((bsz, 1, 1, 1), device=dev))
        return (x * dim).clamp(0, 1)

    x0, x1 = jit(x0, torch.zeros_like(ident)), jit(x1, ident)
    maps = [x0, x1, h0, g0, m0, h1, g1, m1, rc, uc]
    fl = torch.rand((bsz, 2), device=dev, generator=gen) < 0.5
    rot = torch.randint(0, 4, (bsz,), device=dev, generator=gen) if rotate else torch.zeros(bsz, dtype=torch.long)
    out = [[] for _ in maps]
    for i in range(bsz):
        for j, z in enumerate(maps):
            v = z[i]
            if fl[i, 0]:
                v = v.flip(-1)
            if fl[i, 1]:
                v = v.flip(-2)
            if rot[i]:
                v = torch.rot90(v, int(rot[i]), dims=(-2, -1))
            out[j].append(v)
    x0, x1, h0, g0, m0, h1, g1, m1, rc, uc = (torch.stack(o) for o in out)
    return x0, x1, h0, g0, m0, h1, g1, m1, rc, uc, meta
