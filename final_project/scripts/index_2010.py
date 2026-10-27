"""Tile index for the 2010 PASDA LiDAR (files are named Tile000001.zip etc. with no coordinates).

For each zip, fetch the first 16 KB, inflate the start of the LAS member and read the header extent.
Writes data/raw/lidar/tileindex2010.csv (name, bytes, minx, miny, maxx, maxy). Resumable.
"""

import re
import struct
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tiles as T  # noqa: E402

OUT = T.LIDAR / "tileindex2010.csv"
html = (T.LIDAR / "pasda_LAS2010_listing.html").read_text()
rows = re.findall(r"(\d+) <A HREF=\"[^\"]*/(Tile\d+)\.zip\"", html)
done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["name"])


def one(args):
    size, name = args
    if name in set(done.name):
        return None
    for _ in range(4):
        try:
            b = requests.get(T.URL2010.format(name=name), headers={"Range": "bytes=0-16383"}, timeout=(10, 30)).content
            nlen, xlen = struct.unpack("<HH", b[26:30])
            raw = zlib.decompressobj(-15).decompress(b[30 + nlen + xlen:])
            assert raw[:4] == b"LASF"
            maxx, minx, maxy, miny = struct.unpack("<4d", raw[179:211])
            return {"name": name, "bytes": int(size), "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
        except Exception as e:  # noqa: BLE001
            err = e
    print(f"failed {name}: {err!r}", flush=True)
    return None


new = []
with ThreadPoolExecutor(3) as ex:
    for k, r in enumerate(ex.map(one, rows), 1):
        if r:
            new.append(r)
        if k % 100 == 0:
            print(f"{k}/{len(rows)}", flush=True)
out = pd.concat([done, pd.DataFrame(new)], ignore_index=True)
out.to_csv(OUT, index=False)
print(f"{len(out)} / {len(rows)} tiles indexed")
