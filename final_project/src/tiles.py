"""PASDA LiDAR tiles for a study area: 2018 (zipped LAS, names = lower-left corner), 2022 (LAS + tile index) and
2025 (LAS, names = lower-left corner, 2,640 ft squares).

Coordinates: EPSG:2272 (2018) and EPSG:6565 (2022) are both PA South US-ft; they differ only by NAD83 realisation.
"""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
LIDAR = ROOT / "data" / "raw" / "lidar"
BOUNDARIES = ROOT / "data" / "raw" / "boundaries"
URL2018 = "https://www.pasda.psu.edu/download/phillyLiDAR/LAS2018/{name}.zip"
URL2022 = "https://www.pasda.psu.edu/download/phillyLiDAR/2022/LAS/Classified_LAS/{name}.las"
URL2025 = "https://www.pasda.psu.edu/download/phillyLiDAR/2025/LAS/{name}.las"


def study_area(zips=("19146",), buffer_ft=300) -> gpd.GeoDataFrame:
    z = gpd.read_file(BOUNDARIES / "zip_codes.geojson").to_crs(2272)
    z["zip"] = z["code"].astype(str)
    s = z[z["zip"].isin(zips)][["zip", "geometry"]].copy()
    s["geometry"] = s.buffer(buffer_ft)
    return s


def tiles_2018() -> gpd.GeoDataFrame:
    html = (LIDAR / "pasda_LAS2018_listing.html").read_text()
    rows = re.findall(r"(\d+)\s*<A HREF=\"[^\"]*?/(\d+)E(\d+)N\.zip\"", html, re.I)
    df = pd.DataFrame(rows, columns=["bytes", "e", "n"])
    df["name"] = df.e + "E" + df.n + "N"
    x0, y0 = df.e.astype(int) * 100, df.n.astype(int)
    df["bytes"] = df.bytes.astype(int)
    df["url"] = [URL2018.format(name=n) for n in df.name]
    df["file"] = [LIDAR / "las2018" / f"{n}.zip" for n in df.name]
    # names are rounded down to 100 ft, so pad the box by 100 ft to be safe
    return gpd.GeoDataFrame(df, geometry=[box(x - 100, y, x + 2740, y + 2640) for x, y in zip(x0, y0)], crs=2272)


def tiles_2022() -> gpd.GeoDataFrame:
    t = gpd.read_file(LIDAR / "tileindex2022" / "Philadelphia_Tiles.shp").set_crs(6565, allow_override=True)
    html = (LIDAR / "pasda_LAS2022_listing.html").read_text()
    sizes = {n: int(s) for s, n in re.findall(r"(\d+) <A HREF=\"[^\"]*/(Philadelphia_\d+)\.las\"", html)}
    t["name"] = t.NAME
    t["bytes"] = t.name.map(sizes)
    t["url"] = [URL2022.format(name=n) for n in t.name]
    t["file"] = [LIDAR / "las2022" / f"{n}.las" for n in t.name]
    return t.to_crs(2272)[["name", "bytes", "url", "file", "geometry"]]


def tiles_2025() -> gpd.GeoDataFrame:
    html = (LIDAR / "pasda_LAS2025_listing.html").read_text()
    df = pd.DataFrame(re.findall(r"(\d+) <A HREF=\"[^\"]*/(\d+)E(\d+)N\.las\"", html), columns=["bytes", "e", "n"])
    df["name"] = df.e + "E" + df.n + "N"
    x0, y0 = df.e.astype(int) * 100, df.n.astype(int)
    df["bytes"] = df.bytes.astype(int)
    df["url"] = [URL2025.format(name=n) for n in df.name]
    df["file"] = [LIDAR / "las2025" / f"{n}.las" for n in df.name]
    return gpd.GeoDataFrame(df, geometry=[box(x - 100, y, x + 2740, y + 2640) for x, y in zip(x0, y0)], crs=2272)


URL2015 = "https://www.pasda.psu.edu/download/phillyLiDAR/LAS2015/{name}.zip"
URL2010 = "https://www.pasda.psu.edu/download/phillyLiDAR/LAS2010/{name}.zip"


def tiles_2015() -> gpd.GeoDataFrame:
    """2015: zipped LAS, 2,400 ft tiles named by the lower-left corner in units of 100 ft (e.g. 26670E2316N)."""
    html = (LIDAR / "pasda_LAS2015_listing.html").read_text()
    df = pd.DataFrame(re.findall(r"(\d+) <A HREF=\"[^\"]*/(\d+)E(\d+)N\.zip\"", html), columns=["bytes", "e", "n"])
    df["name"] = df.e + "E" + df.n + "N"
    x0, y0 = df.e.astype(int) * 100, df.n.astype(int) * 100
    df["bytes"] = df.bytes.astype(int)
    df["url"] = [URL2015.format(name=n) for n in df.name]
    df["file"] = [LIDAR / "las2015" / f"{n}.zip" for n in df.name]
    return gpd.GeoDataFrame(df, geometry=[box(x, y, x + 2400, y + 2400) for x, y in zip(x0, y0)], crs=2272)


def tiles_2010() -> gpd.GeoDataFrame:
    """2010: zipped LAS named Tile000001..; extents read from each LAS header (scripts/index_2010.py -> csv)."""
    d = pd.read_csv(LIDAR / "tileindex2010.csv")
    d["url"] = [URL2010.format(name=n) for n in d.name]
    d["file"] = [LIDAR / "las2010" / f"{n}.zip" for n in d.name]
    return gpd.GeoDataFrame(d, geometry=[box(a, b, c, e) for a, b, c, e in zip(d.minx, d.miny, d.maxx, d.maxy)],
                            crs=2272)


def study_tiles(year: int, zips=("19146",)) -> gpd.GeoDataFrame:
    t = {2010: tiles_2010, 2015: tiles_2015, 2018: tiles_2018, 2022: tiles_2022, 2025: tiles_2025}[year]()
    return t[t.intersects(study_area(zips).union_all())].copy()


if __name__ == "__main__":
    for y in (2018, 2022):
        t = study_tiles(y)
        have = sum(f.exists() and f.stat().st_size == b for f, b in zip(t.file, t.bytes))
        print(f"{y}: {len(t)} tiles, {t.bytes.sum() / 1e9:.2f} GB ({'zipped' if y == 2018 else 'uncompressed LAS'}), "
              f"already on disk: {have}")
