"""Unit-size cache and the equivalence check for POPUP_PREP_CACHE=1 (training without the citywide rasters).

    POPUP_DATA=citywide python scripts/prep_cache.py sizes     # write data/interim/unit_ids_city/sizes/{ZIP}.csv
    POPUP_DATA=citywide python scripts/prep_cache.py check     # 50 random train / val pairs -> equivalence_check.md

`check` recomputes, in memory only, what training's prepare() would rebuild from the rasters - the real-change mask
of each pair (bitemporal.change_mask_raster), the unit-id patch and the per-unit cell counts (unitcls.unit_id_grid)
- and compares them with the saved files the cache mode reads. Nothing in the training data folders is written.
Both commands run at idle CPU and very low I/O priority (training runs at the same time).
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bitemporal as B  # noqa: E402
import height_data as H  # noqa: E402
import unitcls as U  # noqa: E402

assert H.CITY and not U.PREP_CACHE, "run with POPUP_DATA=citywide and without POPUP_PREP_CACHE"
PERIODS = ("2010-2015", "2010-2018", "2010-2022", "2015-2018", "2015-2022", "2018-2022")
N_CHECK, SEED, MAX_CELLS = 50, 20261026, 40e6   # windows up to 40 M cells (~1 GB to rebuild): memory is shared


def low_priority():
    p = psutil.Process()
    p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 19)
    try:
        p.ionice(psutil.IOPRIO_VERYLOW if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    except Exception:  # noqa: BLE001
        pass


def wait_memory(gb=2.5):
    while psutil.virtual_memory().available / 1e9 < gb:
        time.sleep(30)


def zips():
    idx = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
    return idx, sorted(idx[idx.split.isin(["train", "val"])].zip.unique())


def sizes():
    _, zs = zips()
    for z in zs:
        wait_memory()
        U.save_unit_sizes(z, U.unit_sizes(U.unit_id_grid(z, 2018)))
        print(f"  sizes {z}", flush=True)


def check():
    idx, zs = zips()
    tv = idx[idx.split.isin(["train", "val"])]
    cells = {}
    for z in zs:
        d = np.load(H.raster_path(z, 2018))
        cells[z] = int(d["nx"]) * int(d["ny"])
    pairs = []
    for per in PERIODS:
        y0, y1 = (int(v) for v in per.split("-"))
        a = tv[tv.year == y0][["zip", "i", "j", "split"]].merge(tv[tv.year == y1][["zip", "i", "j"]])
        pairs.append(a.assign(y0=y0, y1=y1))
    pairs = pd.concat(pairs, ignore_index=True)
    eligible = pairs[pairs.zip.map(cells) <= MAX_CELLS]
    rng = np.random.default_rng(SEED)
    s = eligible.iloc[np.sort(rng.choice(len(eligible), N_CHECK, replace=False))]
    rows, size_rows = [], []
    for z, g in s.groupby("zip"):
        wait_memory()
        uid = U.unit_id_grid(z, 2018)
        rec, cached = U.unit_sizes(uid), U.load_unit_sizes(z)
        size_rows.append({"zip": z, "units": len(rec),
                          "sizes_identical": bool(rec.sort_index().equals(cached.sort_index().astype(rec.dtype)))})
        for (y0, y1), gg in g.groupby(["y0", "y1"]):
            mk = B.change_mask_raster(z, y0, y1)
            for r in gg.itertuples():
                sl = np.s_[r.i * H.P:(r.i + 1) * H.P, r.j * H.P:(r.j + 1) * H.P]
                m_file = np.load(B.MASK_DIR / f"{z}_{y0}_{y1}_{r.i}_{r.j}.npy")
                u_file = np.load(U.UID_DIR / f"{z}_{r.i}_{r.j}.npy")
                rows.append({"zip": z, "period": f"{y0}-{y1}", "i": r.i, "j": r.j, "split": r.split,
                             "mask_cells": int(m_file.sum()), "mask_identical": bool(np.array_equal(mk[sl], m_file)),
                             "uid_identical": bool(np.array_equal(uid[sl], u_file))})
            del mk
        del uid
        print(f"  checked {z}", flush=True)
    r, sz = pd.DataFrame(rows), pd.DataFrame(size_rows)
    ok = r.mask_identical.all() and r.uid_identical.all() and sz.sizes_identical.all()
    out = ROOT / "results/citywide/m4/equivalence_check.md"
    out.write_text(
        "# Equivalence check: saved masks / unit ids (POPUP_PREP_CACHE=1) vs rebuilding from the rasters\n\n"
        f"Run {time.strftime('%Y-%m-%d %H:%M')} by `scripts/prep_cache.py check` (seed {SEED}).\n\n"
        f"- Sample: {len(r)} random train / val pairs over all six periods, from ZIP windows up to "
        f"{MAX_CELLS / 1e6:.0f} M cells (rebuilding a larger window needs several GB of memory while training "
        f"runs); {r.zip.nunique()} ZIPs, {r.period.nunique()} periods, {int((r.split == 'val').sum())} val pairs, "
        f"{int((r.mask_cells > 0).sum())} pairs with real change.\n"
        "- Compared exactly (every cell): the real-change mask of the pair, the unit-id patch, and the cells per "
        "unit over the whole ZIP grid (used to decide whether a unit lies >= 90% inside a crop).\n\n"
        f"**Result: {'PASS - all identical' if ok else 'FAIL'}**\n\n"
        f"| | identical |\n|---|---|\n| masks | {int(r.mask_identical.sum())} / {len(r)} |\n"
        f"| unit-id patches | {int(r.uid_identical.sum())} / {len(r)} |\n"
        f"| unit sizes (per ZIP) | {int(sz.sizes_identical.sum())} / {len(sz)} ZIPs |\n\n"
        "Per pair:\n\n" + r.to_markdown(index=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if ok else 'FAIL'}: masks {r.mask_identical.sum()}/{len(r)}, uids {r.uid_identical.sum()}/"
          f"{len(r)}, sizes {sz.sizes_identical.sum()}/{len(sz)}", flush=True)


if __name__ == "__main__":
    low_priority()
    {"sizes": sizes, "check": check}[sys.argv[1]]()
