"""Pack the M4 training data and code for cloud training (no upload). Local tool; see the package README.md.

    POPUP_DATA=citywide python cloud/pack.py OUT_DIR [--zstd-path DIR]

Whitelist only: the train / val patches of data/interim/m4_patches.csv and, for them, the training inputs (.npy),
per-group index.csv, real-change masks of the six training periods, unit-id patches and unit sizes, and the LiDAR
labels of the six periods. Never packed: anything of the eight sealed test ZIPs, the citywide rasters, raw LAS,
the imagery cache, unit polygons (they carry addresses), review pages and labels.
data/: zstd tar shards of about 2 GB (level 3, 2 threads) + the DINOv3 weights uncompressed + MANIFEST.csv +
SHA256SUMS. code/: code.tar.zst (selected files of `git archive HEAD`) + SHA256SUMS. Runs at idle CPU and very
low I/O priority. Writes exclusion_check.md (geometry of every packed patch vs the test ZIPs + 100 m).
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--zstd-path", help="folder holding the zstandard package (kept out of the project venv)")
ap.add_argument("--code-only", action="store_true",
                help="rebuild code/ and the exclusion check only; data/ (shards, MANIFEST.csv) is reused as it is")
a = ap.parse_args()
if a.zstd_path:
    sys.path.insert(0, a.zstd_path)
import zstandard  # noqa: E402

import height_data as H  # noqa: E402
import tiles as T  # noqa: E402
from rasters import FT  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
OUT = Path(a.out)
PERIODS = ("2010-2015", "2010-2018", "2010-2022", "2015-2018", "2015-2022", "2018-2022")
SHARD_BYTES = 2 * 1000 ** 3
TEST = ("19106", "19116", "19127", "19134", "19136", "19139", "19150", "19153")
RAS = ROOT / "data/interim/citywide_rasters"


def low_priority():
    p = psutil.Process()
    p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    try:
        p.ionice(psutil.IOPRIO_VERYLOW if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    except Exception:  # noqa: BLE001
        pass


def sha256_file(f, chunk=1 << 22):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


class Counting:
    """File wrapper that counts and hashes the compressed bytes as they are written."""

    def __init__(self, f):
        self.f, self.n, self.h = f, 0, hashlib.sha256()

    def write(self, b):
        self.f.write(b)
        self.n += len(b)
        self.h.update(b)
        return len(b)

    def flush(self):
        self.f.flush()


class Shards:
    def __init__(self, folder):
        self.folder, self.k, self.rows, self.sums = folder, 0, [], []
        self.cctx = zstandard.ZstdCompressor(level=3, threads=2)
        self._open()

    def _open(self):
        self.name = f"shard_{self.k:03d}.tar.zst"
        self.raw = open(self.folder / self.name, "wb")
        self.cnt = Counting(self.raw)
        self.zw = self.cctx.stream_writer(self.cnt, closefd=False)
        self.tar = tarfile.open(fileobj=self.zw, mode="w|")

    def _close(self):
        self.tar.close()
        self.zw.close()
        self.raw.close()
        self.sums.append((self.cnt.h.hexdigest(), self.name, self.cnt.n))

    def add_bytes(self, arcname, data):
        if self.cnt.n >= SHARD_BYTES:
            self._close()
            self.k += 1
            self._open()
        ti = tarfile.TarInfo(arcname)
        ti.size, ti.mtime = len(data), int(time.time())
        self.tar.addfile(ti, io.BytesIO(data))
        self.rows.append({"shard": self.name, "path": arcname, "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest()})

    def add_file(self, path):
        self.add_bytes(path.relative_to(ROOT).as_posix(), path.read_bytes())

    def finish(self):
        self._close()


def main():
    low_priority()
    data_dir, code_dir = OUT / "data", OUT / "code"
    data_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
    tv = idx[idx.split.isin(["train", "val"])].copy()
    assert not set(tv.zip) & set(TEST)
    zips = sorted(tv.zip.unique())

    # pair index (item 3): one row per patch pair and period
    blk = 1000 / FT
    side = H.P * H.CELL
    pairs = []
    for per in PERIODS:
        y0, y1 = (int(v) for v in per.split("-"))
        p = tv[tv.year == y0][["zip", "i", "j", "x0", "y1", "split"]].merge(tv[tv.year == y1][["zip", "i", "j"]])
        pairs.append(p.assign(period=per, y0=y0, y1_year=y1))
    pi = pd.concat(pairs, ignore_index=True)
    pi["patch_id"] = pi.zip + "_" + pi.i.astype(str) + "_" + pi.j.astype(str)
    pi = pi.rename(columns={"x0": "xmin_ft", "y1": "ymax_ft"})
    pi["xmax_ft"], pi["ymin_ft"] = pi.xmin_ft + side, pi.ymax_ft - side
    cx, cy = pi.xmin_ft + side / 2, pi.ymax_ft - side / 2
    pi["block_1km"] = [f"{int(x // blk)}_{int(y // blk)}" for x, y in zip(cx, cy)]

    keep = ["unit", "n_cells", "cover18", "cover22", "bldg_m2_18", "bldg_m2_22", "h50_18", "h90_18", "h50_22",
            "h90_22", "n_dh_cells", "frac_up", "frac_down", "dh_p50", "new_outside_fp_m2", "veg_share_18",
            "veg_share_22", "data_18", "data_22", "chg_new", "chg_popup", "chg_demo", "in_mask", "area_type"]
    if a.code_only:
        man = pd.read_csv(data_dir / "MANIFEST.csv")
        st = pd.read_parquet(ROOT / "data/interim/lidar_change_city" / f"{zips[0]}_2018_2022.parquet")[keep]
        code_and_check(pi, man, st, code_dir, t0)
        return
    sh = Shards(data_dir)
    # 1 training inputs (.npy) and per-group index.csv
    n_patch = 0
    for (z, y), g in tv.groupby(["zip", "year"]):
        sh.add_file(H.OUT / f"{z}_{y}" / "index.csv")
        for i, j in zip(g.i, g.j):
            stem = f"{i}_{j}"
            for k in ("img", "ndsm", "veg", "ndsm_filled"):
                f = ROOT / "data/interim/height_npy_city" / f"{z}_{y}" / f"{stem}_{k}.npy"
                if f.exists():
                    sh.add_file(f)
                elif k != "ndsm_filled":
                    raise FileNotFoundError(f)
            n_patch += 1
        print(f"  patches {z} {y}: {len(g)} ({n_patch}/{len(tv)}) shard {sh.k} {time.time() - t0:.0f}s", flush=True)
    # masks (+ n_real for the pair index)
    n_real = []
    for r in pi.itertuples():
        f = ROOT / "data/interim/bitemporal_masks_city" / f"{r.zip}_{r.y0}_{r.y1_year}_{r.i}_{r.j}.npy"
        b = f.read_bytes()
        sh.add_bytes(f.relative_to(ROOT).as_posix(), b)
        n_real.append(int(np.load(io.BytesIO(b)).sum()))
    pi["n_real_cells"] = n_real
    print(f"  masks {len(pi)} {time.time() - t0:.0f}s", flush=True)
    # unit ids and unit sizes
    for z, i, j in tv[["zip", "i", "j"]].drop_duplicates().itertuples(index=False):
        sh.add_file(ROOT / "data/interim/unit_ids_city" / f"{z}_{i}_{j}.npy")
    for z in zips:
        sh.add_file(ROOT / "data/interim/unit_ids_city/sizes" / f"{z}.csv")
    # labels of the six training periods
    for z in zips:
        for per in PERIODS:
            sh.add_file(ROOT / "data/interim/lidar_change_city" / f"{z}_{per.replace('-', '_')}.parquet")
    # per-unit change statistics (item 2), exported from the per-unit results
    st = pd.concat([pd.read_parquet(ROOT / "data/interim/lidar_change_city" / f"{z}_{per.replace('-', '_')}.parquet")
                    [keep].assign(zip=z, period=per) for z in zips for per in PERIODS], ignore_index=True)
    # the change features name the two epochs 18 / 22 for historical reasons: here they are the period's y0 / y1
    st = st.rename(columns={c: c[:-2].rstrip("_") + ("_y0" if c.endswith("18") else "_y1")
                            for c in keep if c.endswith(("18", "22"))})
    buf = io.BytesIO()
    st.to_parquet(buf, index=False)
    sh.add_bytes("data/interim/m4_unit_change_stats.parquet", buf.getvalue())
    # pair index, patch selection (train / val only) and split blocks (items 3, 4)
    cols = ["patch_id", "zip", "i", "j", "period", "split", "block_1km", "xmin_ft", "ymin_ft", "xmax_ft", "ymax_ft",
            "n_real_cells"]
    sh.add_bytes("data/interim/m4_pair_index.csv", pi[cols].to_csv(index=False).encode())
    sh.add_bytes("data/interim/m4_patches.csv", tv.to_csv(index=False).encode())
    bl = pd.read_csv(ROOT / "results/citywide/splits_citywide.csv")
    sh.add_bytes("data/splits/splits_citywide_trainval_blocks.csv",
                 bl[bl.split.isin(["train", "val"])].to_csv(index=False).encode())
    sh.finish()
    print(f"  data shards: {sh.k + 1}, {time.time() - t0:.0f}s", flush=True)

    # weights, uncompressed
    w = next((ROOT / "data/interim/models").glob("dinov3_vitl16_pretrain_sat493m-*.pth"))
    wdst = data_dir / w.name
    if not (wdst.exists() and wdst.stat().st_size == w.stat().st_size):
        with open(w, "rb") as src, open(wdst, "wb") as dst:
            while b := src.read(1 << 22):
                dst.write(b)
    man = pd.DataFrame(sh.rows)
    man = pd.concat([man, pd.DataFrame([{"shard": w.name, "path": f"data/interim/models/{w.name}",
                                         "bytes": w.stat().st_size, "sha256": sha256_file(wdst)}])])
    man.to_csv(data_dir / "MANIFEST.csv", index=False)
    with open(data_dir / "SHA256SUMS", "w", newline="\n") as fh:
        for h, n, _ in sh.sums:
            fh.write(f"{h}  {n}\n")
        fh.write(f"{man.sha256.iloc[-1]}  {w.name}\n")
    sh_sizes = pd.DataFrame(sh.sums, columns=["sha256", "shard", "bytes"])
    sh_sizes.to_csv(data_dir / "SHARDS.csv", index=False)
    code_and_check(pi, man, st, code_dir, t0)


def code_and_check(pi, man, st, code_dir, t0):
    # code archive: selected files of git archive HEAD
    # core.autocrlf=false: on Windows git archive would otherwise write CRLF, which bash on Linux cannot run
    raw = subprocess.run(["git", "-c", "core.autocrlf=false", "archive", "--format=tar", "HEAD", "--", "."],
                         cwd=ROOT, capture_output=True, check=True).stdout
    keep_prefix = ("src/", "scripts/", "cloud/")
    keep_files = {"pyproject.toml", "uv.lock", ".python-version", "README.md", "results/citywide/M4_plan.md",
                  "results/citywide/splits_citywide.md", "results/citywide/study_area.md",
                  "results/citywide/veg2025_fix.md", "results/citywide/m4/queue_m4.sh",
                  "results/citywide/m4/queue_a1.sh", "results/citywide/m4/equivalence_check.md"}
    src_tar = tarfile.open(fileobj=io.BytesIO(raw), mode="r:")
    cbuf = io.BytesIO()
    kept = []
    with tarfile.open(fileobj=cbuf, mode="w") as out:
        for m in src_tar.getmembers():
            if not m.isfile():
                continue
            if m.name.startswith(keep_prefix) or m.name in keep_files:
                assert not m.name.endswith((".html", ".csv", ".parquet", ".npz", ".npy")), m.name  # code only
                m2 = tarfile.TarInfo("final_project/" + m.name)
                m2.size, m2.mtime, m2.mode = m.size, m.mtime, m.mode
                out.addfile(m2, src_tar.extractfile(m))
                kept.append(m.name)
    missing = keep_files - set(kept)
    assert not missing, f"not in HEAD: {missing}"
    (code_dir / "code.tar.zst").write_bytes(zstandard.ZstdCompressor(level=19, threads=2).compress(cbuf.getvalue()))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (code_dir / "FILES.txt").write_text(f"git HEAD {head}\n" + "\n".join(sorted(kept)) + "\n", encoding="utf-8")
    for extra in ("cloud/setup_cloud.sh", "cloud/sync_back.txt", "results/citywide/M4_plan.md",
                  "results/citywide/m4/equivalence_check.md"):
        (code_dir / Path(extra).name).write_bytes((ROOT / extra).read_bytes().replace(b"\r\n", b"\n"))
    with open(code_dir / "SHA256SUMS", "w", newline="\n") as fh:
        for f in sorted(code_dir.iterdir()):
            if f.name != "SHA256SUMS":
                fh.write(f"{sha256_file(f)}  {f.name}\n")

    # exclusion check
    tz = T.study_area(TEST, buffer_ft=0).union_all()
    tb = T.study_area(TEST, buffer_ft=100 / FT).union_all()
    from shapely.geometry import box
    g = [box(r.xmin_ft, r.ymin_ft, r.xmax_ft, r.ymax_ft) for r in pi.drop_duplicates("patch_id").itertuples()]
    hit0, hit100 = sum(b.intersects(tz) for b in g), sum(b.intersects(tb) for b in g)
    bad_names = man[man.path.str.contains("|".join(TEST))]
    forbidden = man[man.path.str.contains("citywide_rasters|citywide_tiles|raw/|ortho|review|citywide_units|label",
                                          case=False)]
    addr_cols = [c for c in st.columns if "addr" in c.lower()] + [c for c in pi.columns if "addr" in c.lower()]
    ok = hit0 == 0 and hit100 == 0 and bad_names.empty and forbidden.empty and not addr_cols
    (OUT / "exclusion_check.md").write_text(
        "# Exclusion check of the cloud package\n\n"
        f"Run {time.strftime('%Y-%m-%d %H:%M')} by `cloud/pack.py` on the packed files (git HEAD {head[:7]}).\n\n"
        f"**Result: {'PASS' if ok else 'FAIL'}**\n\n"
        "| Check | Result |\n|---|---|\n"
        f"| Packed patches (unique) | {len(g):,} in {pi.zip.nunique()} ZIPs; {len(pi):,} pairs over 6 periods |\n"
        f"| Patches intersecting a sealed test ZIP | {hit0} |\n"
        f"| Patches intersecting a test ZIP buffered by 100 m | {hit100} |\n"
        f"| Packed file paths naming a test ZIP ({', '.join(TEST)}) | {len(bad_names)} |\n"
        f"| Packed paths from rasters, LAS, imagery, unit polygons, review pages or labels | {len(forbidden)} |\n"
        f"| Address columns in the exported tables | {len(addr_cols)} |\n\n"
        "Geometry: each patch box from `m4_pair_index.csv` (EPSG:2272 ft) against the 2019 ZIP polygons of the "
        "eight test ZIPs, unbuffered and buffered by 100 m.\n", encoding="utf-8")
    print(f"exclusion check {'PASS' if ok else 'FAIL'}; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
