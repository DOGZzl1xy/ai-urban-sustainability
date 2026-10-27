"""0.5 m DSM / DTM / vegetation rasters for one LiDAR epoch over the study area, built tile by tile.

- DSM: highest single-return point per cell (classes 7, 12, 18 = noise / overlap dropped; multi-return = vegetation).
- DTM: lowest ground (class 2) point per 1.5 m cell, gaps filled by linear interpolation, upsampled to 0.5 m.
- veg: share of returns from multi-return pulses in the cell (used to mask cells where trees dominate).
  The 2022 "Classified_LAS" only separates ground (no vegetation / building classes), so vegetation is detected the
  same way in both epochs from return structure: leaf-off canopy splits pulses into several returns, roofs do not.
Each epoch's nDSM uses its own DTM, so a vertical datum offset between epochs cancels out of ΔnDSM.
"""

from __future__ import annotations

import zipfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import zoom

FT = 0.3048
CELL = 0.5 / FT          # 0.5 m in US ft
DTM_FACTOR = 3           # DTM cell = 1.5 m
DROP = (7, 12, 18)


@dataclass
class Grid:
    x0: float
    y1: float            # top edge (row 0 is north)
    nx: int
    ny: int

    @classmethod
    def around(cls, bounds, pad_ft=50.0):
        x0, y0, x1, y1 = bounds
        x0, y0, x1, y1 = x0 - pad_ft, y0 - pad_ft, x1 + pad_ft, y1 + pad_ft
        return cls(x0, y1, int(np.ceil((x1 - x0) / CELL)), int(np.ceil((y1 - y0) / CELL)))

    def index(self, x, y):
        return ((self.y1 - y) / CELL).astype(np.int64), ((x - self.x0) / CELL).astype(np.int64)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.x0, self.y1, CELL, CELL)


def read_points(path: Path):
    if path.suffix == ".zip":
        with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".las", ".laz")))
            z.extract(name, td)
            las = laspy.read(Path(td) / name)
    else:
        try:
            las = laspy.read(path)
        except ValueError:
            # some 2025 PASDA files are truncated at the source (shorter than the header says): keep whole points
            with laspy.open(path) as r:
                avail = (path.stat().st_size - r.header.offset_to_point_data) // r.header.point_format.size
                las = laspy.LasData(r.header, r.read_points(int(avail)))
            print(f"  warning: {path.name} truncated, read {avail:,} of {r.header.point_count:,} points", flush=True)
    cls = np.asarray(las.classification)
    keep = ~np.isin(cls, DROP)
    if hasattr(las, "withheld"):
        keep &= ~np.asarray(las.withheld, dtype=bool)
    multi = np.asarray(las.number_of_returns) > 1
    return np.asarray(las.x)[keep], np.asarray(las.y)[keep], np.asarray(las.z)[keep], cls[keep], multi[keep]


def tile_rasters(path: Path, grid: Grid, veg_classes=()):
    """Per-cell DSM max, ground min (on the 1.5 m grid), and vegetation / total counts for one tile.
    veg_classes: ASPRS classes also counted as vegetation (2025: 3, 4, 5). The 2025 flight (18 April, leaves coming
    out) has 42-58% of its high-vegetation points as single returns, so return structure alone misses canopy."""
    x, y, z, cls, multi = read_points(path)
    if veg_classes:
        multi = multi | np.isin(cls, veg_classes)
    r, c = grid.index(x, y)
    ok = (r >= 0) & (r < grid.ny) & (c >= 0) & (c < grid.nx)
    x, y, z, cls, multi, r, c = x[ok], y[ok], z[ok], cls[ok], multi[ok], r[ok], c[ok]
    if len(z) == 0:
        return None
    r0, r1, c0, c1 = r.min(), r.max() + 1, c.min(), c.max() + 1
    h, w = r1 - r0, c1 - c0
    flat = (r - r0) * w + (c - c0)
    veg = multi
    dsm = np.full(h * w, -np.inf, np.float32)
    np.maximum.at(dsm, flat[~veg], z[~veg].astype(np.float32))
    nveg = np.bincount(flat[veg], minlength=h * w).astype(np.int32)
    ntot = np.bincount(flat, minlength=h * w).astype(np.int32)
    g = cls == 2
    GR, GC = r[g] // DTM_FACTOR, c[g] // DTM_FACTOR  # global 1.5 m grid indices
    gr0, gc0 = int(r0 // DTM_FACTOR), int(c0 // DTM_FACTOR)
    gh, gw = int((r1 - 1) // DTM_FACTOR) - gr0 + 1, int((c1 - 1) // DTM_FACTOR) - gc0 + 1
    gmin = np.full(gh * gw, np.inf, np.float32)
    np.minimum.at(gmin, (GR - gr0) * gw + (GC - gc0), z[g].astype(np.float32))
    return dict(r0=int(r0), c0=int(c0), dsm=dsm.reshape(h, w), nveg=nveg.reshape(h, w), ntot=ntot.reshape(h, w),
                gr0=gr0, gc0=gc0, gmin=gmin.reshape(gh, gw))


class Mosaic:
    """Accumulates tile rasters into study-area arrays."""

    def __init__(self, grid: Grid):
        self.g = grid
        self.dsm = np.full((grid.ny, grid.nx), -np.inf, np.float32)
        self.nveg = np.zeros((grid.ny, grid.nx), np.int32)
        self.ntot = np.zeros((grid.ny, grid.nx), np.int32)
        gh, gw = (grid.ny + DTM_FACTOR - 1) // DTM_FACTOR, (grid.nx + DTM_FACTOR - 1) // DTM_FACTOR
        self.gmin = np.full((gh, gw), np.inf, np.float32)

    def add(self, t):
        if t is None:
            return
        r0, c0 = t["r0"], t["c0"]
        h, w = t["dsm"].shape
        sl = (slice(r0, r0 + h), slice(c0, c0 + w))
        np.maximum(self.dsm[sl], t["dsm"], out=self.dsm[sl])
        self.nveg[sl] += t["nveg"]
        self.ntot[sl] += t["ntot"]
        gr0, gc0 = t["gr0"], t["gc0"]
        gh, gw = t["gmin"].shape
        sl2 = (slice(gr0, gr0 + gh), slice(gc0, gc0 + gw))
        np.minimum(self.gmin[sl2], t["gmin"], out=self.gmin[sl2])

    def finish(self):
        """Fill the DTM, return (dsm, dtm, ndsm in metres, veg_share, has_ground_coarse)."""
        gm = self.gmin
        have = np.isfinite(gm)
        rr, cc = np.nonzero(have)
        gy, gx = np.mgrid[0:gm.shape[0], 0:gm.shape[1]]
        filled = griddata((rr, cc), gm[have], (gy, gx), method="linear")
        hole = ~np.isfinite(filled)
        if hole.any():
            filled[hole] = griddata((rr, cc), gm[have], (gy[hole], gx[hole]), method="nearest")
        dtm = zoom(filled, DTM_FACTOR, order=1)[: self.g.ny, : self.g.nx].astype(np.float32)
        dsm = np.where(np.isfinite(self.dsm), self.dsm, np.nan)
        ndsm = (dsm - dtm) * FT
        veg = np.where(self.ntot > 0, self.nveg / np.maximum(self.ntot, 1), np.nan).astype(np.float32)
        return dsm, dtm, ndsm.astype(np.float32), veg, have
