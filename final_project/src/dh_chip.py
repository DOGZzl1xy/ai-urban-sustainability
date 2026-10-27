"""ΔnDSM (2022 − 2018) heat-map chip around a parcel, parcel outlined in black; red = higher, blue = lower (±6 m)."""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import change as C

_DH = {}


def _load(zip_code="19146"):
    if zip_code not in _DH:
        import height_data as H
        from rasters import Grid

        a, b = np.load(H.raster_path(zip_code, 2018)), np.load(H.raster_path(zip_code, 2022))
        grid = Grid(float(a["x0"]), float(a["y1"]), int(a["nx"]), int(a["ny"]))
        _DH[zip_code] = (b["ndsm"] - a["ndsm"], grid)
    return _DH[zip_code]


def dh_chip(geom, pad_m=10, size=300, zip_code="19146") -> Image.Image:
    dh, grid = _load(zip_code)
    x0, y0, x1, y1 = geom.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 + pad_m / 0.3048
    r0, c0 = grid.index(np.array([cx - half]), np.array([cy + half]))
    r1, c1 = grid.index(np.array([cx + half]), np.array([cy - half]))
    sub = dh[max(r0[0], 0):r1[0], max(c0[0], 0):c1[0]]
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(np.clip(sub, -6, 6), cmap="RdBu_r", vmin=-6, vmax=6, extent=(cx - half, cx + half, cy - half, cy + half))
    g = geom if geom.geom_type == "Polygon" else max(geom.geoms, key=lambda q: q.area)
    ax.plot(*g.exterior.xy, color="k", lw=1)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return Image.open(buf).convert("RGB").resize((size, size))
