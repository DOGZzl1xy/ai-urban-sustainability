"""Small aerial chips from the city's 3-inch tile services (Web Mercator tiles), with the footprint outlined.

Only a handful of buildings are fetched (blind review / case figures); tiles are cached under data/raw/ortho/tiles.
"""

from __future__ import annotations

import io
import os
import threading
import math
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / "ortho" / "tiles"
URL = "https://tiles.arcgis.com/tiles/fLeGjb7u4uXqeF9q/arcgis/rest/services/{service}/MapServer/tile/{z}/{y}/{x}"
SERVICES = {2010: "CityImagery_2010_3in", 2015: "CityImagery_2015_3in", 2016: "CityImagery_2016_3in", 2017: "CityImagery_2017_3in", 2018: "CityImagery_2018_3in",
            2019: "CityImagery_2019_3in", 2020: "CityImagery_2020_3in", 2022: "CityImagery_2022_2in",
            2023: "CityImagery_2023", 2024: "CityImagery_2024_3in", 2025: "CityImagery_2025_3in"}
ORIGIN = 20037508.342787
Z = 21  # ~0.075 m / px, the native 3-inch resolution


def _res(z):
    return 2 * ORIGIN / 256 / 2 ** z


BLANK = (230, 230, 230)


def _fetch(year, z, x, y, f: Path) -> bool:
    """Download one tile; write to a temp file and rename so concurrent readers never see a partial file."""
    f.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = requests.get(URL.format(service=SERVICES[year], z=z, x=x, y=y), timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                tmp = f.with_suffix(f".{threading.get_ident()}.part")
                tmp.write_bytes(r.content)
                os.replace(tmp, f)
                return True
            if r.status_code in (404, 400):
                return False
        except requests.RequestException:
            pass
        time.sleep(2 * (attempt + 1))
    return False


_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def _lock(key):
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def tile(year, z, x, y) -> Image.Image:
    f = CACHE / str(year) / f"{z}_{x}_{y}.jpg"
    for _ in range(2):
        # one thread per tile: on Windows, renaming over a file another thread is writing or reading fails
        with _lock((year, z, x, y)):
            ok = f.exists() or _fetch(year, z, x, y, f)
        if not ok:
            return Image.new("RGB", (256, 256), BLANK)
        try:
            with Image.open(f) as im:
                return im.convert("RGB")
        except (OSError, Image.UnidentifiedImageError):
            f.unlink(missing_ok=True)  # corrupt cache entry: drop it and download again
    return Image.new("RGB", (256, 256), BLANK)


def chip(geom_2272, year, pad_m=10.0, z=Z, size=360) -> Image.Image:
    """Chip around a footprint (EPSG:2272 geometry), footprint outlined in yellow."""
    g = gpd.GeoSeries([geom_2272], crs=2272).to_crs(3857).iloc[0]
    x0, y0, x1, y1 = g.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 + pad_m
    bx0, by0, bx1, by1 = cx - half, cy - half, cx + half, cy + half
    r = _res(z)
    tx0, tx1 = int((bx0 + ORIGIN) / (256 * r)), int((bx1 + ORIGIN) / (256 * r))
    ty0, ty1 = int((ORIGIN - by1) / (256 * r)), int((ORIGIN - by0) / (256 * r))
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic.paste(tile(year, z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = tx0 * 256 * r - ORIGIN, ORIGIN - ty0 * 256 * r  # world coords of mosaic top-left

    def px(x, y):
        return (x - ox) / r, (oy - y) / r

    left, top = px(bx0, by1)
    right, bottom = px(bx1, by0)
    img = mosaic.crop((int(left), int(top), int(right), int(bottom)))
    draw = ImageDraw.Draw(img)
    polys = [g] if g.geom_type == "Polygon" else list(g.geoms)
    for p in polys:
        pts = [(px(x, y)[0] - int(left), px(x, y)[1] - int(top)) for x, y in p.exterior.coords]
        draw.line(pts, fill=(255, 220, 0), width=2)
    return img.resize((size, size), Image.LANCZOS)


def chip_georef(geom_2272, year, pad_m=12.0, out_res_m=0.15):
    """Unannotated chip around a parcel at ``out_res_m`` m/px, plus its EPSG:3857 bounds (x0, y0, x1, y1).

    Used for the vehicle detector: pixel (col, row) maps to x0 + col * res, y1 - row * res in Web Mercator.
    """
    g = gpd.GeoSeries([geom_2272], crs=2272).to_crs(3857).iloc[0]
    x0, y0, x1, y1 = g.bounds
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 + pad_m
    bx0, by0, bx1, by1 = cx - half, cy - half, cx + half, cy + half
    r = _res(Z)
    tx0, tx1 = int((bx0 + ORIGIN) / (256 * r)), int((bx1 + ORIGIN) / (256 * r))
    ty0, ty1 = int((ORIGIN - by1) / (256 * r)), int((ORIGIN - by0) / (256 * r))
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic.paste(tile(year, Z, tx, ty), ((tx - tx0) * 256, (ty - ty0) * 256))
    ox, oy = tx0 * 256 * r - ORIGIN, ORIGIN - ty0 * 256 * r
    img = mosaic.crop((int((bx0 - ox) / r), int((oy - by1) / r), int((bx1 - ox) / r), int((oy - by0) / r)))
    n = max(1, int(round(2 * half / out_res_m)))
    return img.resize((n, n), Image.LANCZOS), (bx0, by0, bx1, by1)


def strip(geom_2272, years=(2018, 2022), size=360, gap=6) -> Image.Image:
    ims = [chip(geom_2272, y, size=size) for y in years]
    out = Image.new("RGB", (len(ims) * size + (len(ims) - 1) * gap, size), (255, 255, 255))
    for i, im in enumerate(ims):
        out.paste(im, (i * (size + gap), 0))
    return out
