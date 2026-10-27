"""Paired aerial image / LiDAR nDSM patches for the height-from-imagery model.

A patch is a P x P block of the 0.5 m LiDAR grid (EPSG:2272). The same-year aerial image (city tile service, Web
Mercator) is warped onto exactly that grid, so pixel (r, c) of the image and of the label cover the same ground.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import ortho_chips as O
from rasters import CELL, FT

ROOT = Path(__file__).resolve().parents[1]
# POPUP_DATA=citywide switches every data path to the citywide (M4) versions; Stage 3 files are left untouched.
CITY = os.environ.get("POPUP_DATA") == "citywide"
OUT = ROOT / "data" / "interim" / ("height_city" if CITY else "height")
P = 512                      # patch size in cells (256 m at 0.5 m)
ZOOM = 19                    # 0.30 m Web Mercator = ~0.23 m on the ground at Philadelphia's latitude
TO_3857 = Transformer.from_crs(2272, 3857, always_xy=True)


def raster_path(zip_code: str, year: int) -> Path:
    if CITY:
        return ROOT / f"data/interim/citywide_rasters/{zip_code}_{year}.npz"
    return ROOT / ("data/interim/rasters_{y}.npz".format(y=year) if zip_code == "19146"
                   else f"data/interim/rasters_{zip_code}_{year}.npz")


@dataclass
class Patch:
    i: int
    j: int
    x0: float   # EPSG:2272 ft, left
    y1: float   # top

    @property
    def bounds(self):
        side = P * CELL
        return self.x0, self.y1 - side, self.x0 + side, self.y1


def units_path(zip_code: str) -> Path:
    return ROOT / (f"data/interim/citywide_units/{zip_code}.parquet" if CITY
                   else f"data/interim/units_{zip_code}.parquet")


def change_path(zip_code: str, y0: int, y1: int) -> Path:
    return ROOT / f"data/interim/{'lidar_change_city' if CITY else 'lidar_change'}/{zip_code}_{y0}_{y1}.parquet"


def patches(grid_x0, grid_y1, nx, ny):
    for i in range(ny // P):
        for j in range(nx // P):
            yield Patch(i, j, grid_x0 + j * P * CELL, grid_y1 - i * P * CELL)


def image_for(patch: Patch, year: int, out_res_ft: float = CELL, size: int = P) -> np.ndarray:
    """Aerial RGB warped onto the patch grid (size x size at out_res_ft), uint8 (3, size, size)."""
    x0, y0, x1, y1 = patch.bounds
    xs, ys = TO_3857.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    mx0, mx1, my0, my1 = min(xs), max(xs), min(ys), max(ys)
    r = O._res(ZOOM)
    tx0, tx1 = int((mx0 + O.ORIGIN) / (256 * r)), int((mx1 + O.ORIGIN) / (256 * r))
    ty0, ty1 = int((O.ORIGIN - my1) / (256 * r)), int((O.ORIGIN - my0) / (256 * r))
    mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            t = np.asarray(O.tile(year, ZOOM, tx, ty))
            mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t
    src_t = from_origin(tx0 * 256 * r - O.ORIGIN, O.ORIGIN - ty0 * 256 * r, r, r)
    dst_t = from_origin(x0, y1, out_res_ft, out_res_ft)
    out = np.zeros((3, size, size), np.uint8)
    for b in range(3):
        reproject(mosaic[..., b], out[b], src_transform=src_t, src_crs="EPSG:3857", dst_transform=dst_t,
                  dst_crs="EPSG:2272", resampling=Resampling.bilinear)
    return out


def edge_offset(img: np.ndarray, ndsm: np.ndarray, max_px: int = 6):
    """Shift (dr, dc) in cells that best aligns image edges with nDSM edges (normalised cross-correlation of
    gradient magnitudes over integer shifts). Positive dr means the image content sits dr cells lower (south)."""
    g = img.astype(np.float32).mean(0)
    h = np.nan_to_num(np.clip(ndsm, 0, 30))

    def grad(a):
        gy, gx = np.gradient(a)
        m = np.hypot(gx, gy)
        return (m - m.mean()) / (m.std() + 1e-6)

    a, b = grad(g), grad(h)
    best, arg = -np.inf, (0, 0)
    c = slice(max_px, -max_px)
    for dr in range(-max_px, max_px + 1):
        for dc in range(-max_px, max_px + 1):
            s = (np.roll(np.roll(a, -dr, 0), -dc, 1)[c, c] * b[c, c]).mean()
            if s > best:
                best, arg = s, (dr, dc)
    return arg, float(best)


FT_PER_M = 1 / FT
