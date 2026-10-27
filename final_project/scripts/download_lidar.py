"""Download PASDA LiDAR tiles for some ZIP codes (resumable, 4 connections). Logs to data/raw/lidar/DOWNLOADS.csv.

    python scripts/download_lidar.py                          # 19146, 2018 + 2022 (feasibility test)
    python scripts/download_lidar.py 2018 19132 19121 19143 19125
"""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import tiles as T  # noqa: E402

LOG = T.LIDAR / "DOWNLOADS.csv"
LOCK = threading.Lock()


def fetch(year, r):
    f = Path(r.file)
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists() and f.stat().st_size == r.bytes:
        return None
    for attempt in range(6):
        try:
            pos = f.stat().st_size if f.exists() else 0
            hdr = {"User-Agent": "Mozilla/5.0", **({"Range": f"bytes={pos}-"} if pos else {})}
            with requests.get(r.url, headers=hdr, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(f, "ab" if pos and resp.status_code == 206 else "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
            if f.stat().st_size == r.bytes:
                break
        except requests.RequestException as e:
            print(f"  retry {r.name}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    with LOCK, open(LOG, "a") as fh:
        fh.write(f"{year},{r.name},{r.url},{f.stat().st_size},{time.strftime('%Y-%m-%d %H:%M')}\n")
    return f"{year} {r.name} {f.stat().st_size / 1e6:.0f} MB"


if __name__ == "__main__":
    if not LOG.exists():
        LOG.write_text("year,tile,url,bytes,downloaded_at\n")
    years = [int(a) for a in sys.argv[1:] if len(a) == 4] or [2018, 2022]
    zips = tuple(a for a in sys.argv[1:] if len(a) == 5) or ("19146",)
    jobs = [(y, r) for y in years for r in T.study_tiles(y, zips).itertuples()]
    with ThreadPoolExecutor(4) as ex:
        for i, msg in enumerate(ex.map(lambda a: fetch(*a), jobs), 1):
            if msg:
                print(f"{i}/{len(jobs)} {msg}", flush=True)
    print("done")
