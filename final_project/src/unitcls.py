"""Stage 3 v3: unit-level change classification on top of the LiDAR-anchored bi-temporal model.

Pixel regression alone is pulled towards "no change" (most pixels do not change), while the decision that matters
is per parcel unit. The anchored model's last feature map is mean-pooled over each unit inside the crop and two
binary heads predict
    grew   = LiDAR flags the unit as new construction or pop-up (Stage 1 rules)
    popup  = LiDAR flags the unit as pop-up
with a focal loss; the anchored height regression stays as an auxiliary loss.
Labels: Stage 1 LiDAR flags of the training ZIPs (data/interim/lidar_change/{zip}_{y0}_{y1}.parquet).
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio import features as rfeat

import bitemporal as B
import change as C
import height_data as H
import height_model as M
from rasters import Grid

ROOT = Path(__file__).resolve().parents[1]
UID_DIR = ROOT / ("data/interim/unit_ids_city" if H.CITY else "data/interim/unit_ids")
MIN_IN_CROP = 0.9      # a unit is scored in a crop only if >= 90% of its cells are inside it
# POPUP_PREP_CACHE=1: training reads the saved unit-id patches and unit sizes (and bitemporal.build_masks the saved
# masks) instead of rebuilding them from the citywide rasters, so the rasters are not needed (cloud training).
PREP_CACHE = os.environ.get("POPUP_PREP_CACHE") == "1"
SIZES_DIR = UID_DIR / "sizes"


# ---------------------------------------------------------------- unit rasters and labels
def unit_id_grid(zip_code, year_grid=2018):
    """Unit ids (+1; 0 = no unit) over the ZIP's whole LiDAR grid, unit footprints inset by 0.25 m."""
    z = np.load(H.raster_path(zip_code, year_grid))
    grid = Grid(float(z["x0"]), float(z["y1"]), int(z["nx"]), int(z["ny"]))
    units = gpd.read_parquet(H.units_path(zip_code))
    g = units.geometry.buffer(-C.INSET_FT)
    ok = ~g.is_empty
    return rfeat.rasterize(zip(g[ok], units.unit[ok] + 1), out_shape=(grid.ny, grid.nx), transform=grid.transform,
                           fill=0, dtype="int32")


def unit_sizes(uid):
    return pd.Series(uid[uid > 0] - 1).value_counts()  # cells per unit over the whole grid


def save_unit_sizes(zip_code, sizes):
    SIZES_DIR.mkdir(parents=True, exist_ok=True)
    sizes.rename_axis("unit").rename("cells").to_csv(SIZES_DIR / f"{zip_code}.csv")


def load_unit_sizes(zip_code):
    s = pd.read_csv(SIZES_DIR / f"{zip_code}.csv", index_col="unit")["cells"]
    s.index = s.index.astype(np.int64)
    return s


def unit_id_patches(zip_code, year_grid=2018):
    """Per-patch rasters of unit ids (+1; 0 = no unit) on the LiDAR grid; returns cells per unit over the grid.
    With POPUP_PREP_CACHE=1 the saved patches and sizes are used as they are (checked to exist)."""
    idx = M.load_index((zip_code,), year_grid)
    if PREP_CACHE:
        miss = [f"{i}_{j}" for i, j in zip(idx.i, idx.j) if not (UID_DIR / f"{zip_code}_{i}_{j}.npy").exists()]
        assert not miss, f"{zip_code}: {len(miss)} unit-id patches missing from the cache"
        return load_unit_sizes(zip_code)
    UID_DIR.mkdir(parents=True, exist_ok=True)
    uid = unit_id_grid(zip_code, year_grid)
    for i, j in zip(idx.i, idx.j):
        np.save(UID_DIR / f"{zip_code}_{i}_{j}.npy", uid[i * H.P:(i + 1) * H.P, j * H.P:(j + 1) * H.P])
    sizes = unit_sizes(uid)
    save_unit_sizes(zip_code, sizes)
    return sizes


def unit_labels(zip_code, y0, y1):
    """unit -> grew / popup / covered (LiDAR data in both years) for one ZIP and period."""
    f = pd.read_parquet(H.change_path(zip_code, y0, y1))
    c0 = 0.3 if y0 < 2018 else 0.95   # sparse 2010 / 2015: about half of the 0.5 m cells hold a point
    c1 = 0.3 if y1 < 2018 else 0.95
    lab = pd.DataFrame({"unit": f.unit, "grew": (f.chg_new | f.chg_popup).astype(np.float32),
                        "popup": f.chg_popup.astype(np.float32),
                        "covered": ((f.data_18 > c0) & (f.data_22 > c1) & f.cover18.notna() & f.cover22.notna()).to_numpy()})
    return lab.set_index("unit")


# ---------------------------------------------------------------- model
class UnitCls(B.BiTemporal):
    """Anchored bi-temporal model + per-unit heads (grew, popup)."""

    def __init__(self, random_init=False, feat=256, pool="mean"):
        super().__init__("anchor", feat=feat, random_init=random_init)
        assert pool in ("mean", "meanmax")
        self.pool = pool
        self.unit_proj = nn.Conv2d(64, 64, 1)
        # mean: mean feature + mean anchor height + mean predicted rise (66). meanmax (M4 amendment 2): also the max
        # feature, the max predicted rise and the share of cells predicted to rise >= 2.5 m (132): a rear pop-up
        # covers only part of a unit, which the mean dilutes; the LiDAR rule itself uses the share of roof that rose.
        self.unit_head = nn.Sequential(nn.Linear(64 + 2 if pool == "mean" else 128 + 4, 64), nn.ReLU(),
                                       nn.Linear(64, 2))
        # prior-probability bias (RetinaNet): start at the base rates (grew ~3%, pop-up ~1%) instead of p = 0.5.
        # Without it the rare-positive focal loss pushes every logit far negative in the first steps, the loss
        # saturates and the unit head stops learning (seen in 3 of 4 runs).
        with torch.no_grad():
            self.unit_head[-1].bias.copy_(torch.tensor([np.log(0.03 / 0.97), np.log(0.01 / 0.99)]))

    def forward(self, x0, x1, h0, g0, uid=None):
        b = x0.shape[0]
        size = x0.shape[-1] // 2
        feats = self._tokens(torch.cat([x0, x1]))
        f = []
        for z, pr, rs in zip(feats, self.proj, self.resize):
            a, c = z[:b], z[b:]
            f.append(rs(pr(torch.cat([a, c, c - a], 1))))
        an = torch.stack([h0 / 10, g0.float()], 1)
        f[2] = f[2] + self.anchor_tok(F.adaptive_avg_pool2d(an, f[2].shape[-1]))
        y = self.fuse[3](f[3])
        y = self.fuse[2](y, f[2])
        y = self.fuse[1](y, f[1])
        y = self.fuse[0](y, f[0])
        y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
        s = self.stem(F.interpolate(torch.cat([x0, x1], 1), size=size, mode="area"))
        hid = self.out[1](self.out[0](torch.cat([y, s, self.anchor(an)], 1)))     # (B, 64, H, W)
        out = self.out[2](hid)
        pix = torch.cat([F.relu(h0[:, None] + out[:, :1]), out[:, 1:]], 1)
        if uid is None:
            return pix, None
        return pix, self.pool_units(self.unit_proj(hid), uid, h0, pix[:, 0])

    def pool_units(self, hid, uid, h0, h1p):
        """Mean feature per (sample, unit) + mean anchor height and mean predicted rise; returns logits and keys."""
        bsz, ch = hid.shape[:2]
        key = uid + (torch.arange(bsz, device=uid.device)[:, None, None] * 10_000_000)  # unique per sample
        m = uid > 0
        k = key[m]
        uniq, inv = torch.unique(k, return_inverse=True)
        n = torch.zeros(len(uniq), device=hid.device).index_add_(0, inv, torch.ones_like(inv, dtype=torch.float))
        fe = hid.permute(0, 2, 3, 1)[m]
        pooled = torch.zeros(len(uniq), ch, device=hid.device, dtype=fe.dtype).index_add_(0, inv, fe) / n[:, None]
        hs = torch.zeros(len(uniq), device=hid.device).index_add_(0, inv, h0[m].float()) / n
        r = (h1p - h0)[m].float()
        rise = torch.zeros(len(uniq), device=hid.device).index_add_(0, inv, r) / n
        x = [pooled.float(), hs[:, None] / 10, rise[:, None] / 3]
        if self.pool == "meanmax":
            fmax = torch.zeros(len(uniq), ch, device=hid.device, dtype=fe.dtype).scatter_reduce_(
                0, inv[:, None].expand(-1, ch), fe, "amax", include_self=False)
            rmax = torch.zeros(len(uniq), device=hid.device).scatter_reduce_(0, inv, r, "amax", include_self=False)
            up = torch.zeros(len(uniq), device=hid.device).index_add_(0, inv, (r >= 2.5).float()) / n
            x += [fmax.float(), rmax[:, None] / 3, up[:, None]]
        logits = self.unit_head(torch.cat(x, 1))
        return logits, uniq, n


def focal_bce(logit, target, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logit)
    ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    w = alpha * target + (1 - alpha) * (1 - target)
    return (w * (1 - pt) ** gamma * ce).mean()
