"""QC for the pop-up feasibility test.

Usage: python qc.py dates   -> lidar_dates.json (acquisition dates from GPS time; run once tiles are on disk)
       python qc.py report  -> qc.md (needs the rasters too)
"""

import json
import tempfile
import zipfile
import sys
from pathlib import Path

import laspy
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import permits as P  # noqa: E402
import rasters as R  # noqa: E402
import tiles as T  # noqa: E402

OUT = ROOT / "results/popup_feasibility"
GPS_EPOCH = pd.Timestamp("1980-01-06")


def epoch_info(year):
    t = T.study_tiles(year)
    t = t[[f.exists() and f.stat().st_size == b for f, b in zip(t.file, t.bytes)]]
    lo, hi, classes, dens, truncated = [], [], {}, [], []
    fmt = None
    for f in t.file:
        if f.suffix == ".zip":
            with tempfile.TemporaryDirectory() as td, zipfile.ZipFile(f) as zf:
                n = next(n for n in zf.namelist() if n.lower().endswith(".las"))
                zf.extract(n, td)
                las = laspy.read(Path(td) / n)
        else:
            try:
                las = laspy.read(f)
            except ValueError:  # file shorter than its header says (some 2025 tiles on PASDA): keep whole points
                with laspy.open(f) as r:
                    avail = (f.stat().st_size - r.header.offset_to_point_data) // r.header.point_format.size
                    las = laspy.LasData(r.header, r.read_points(int(avail)))
                truncated.append(f"{f.name}: {avail:,} of {r.header.point_count:,} points")
        g = np.asarray(las.gps_time)
        lo.append(float(g.min())), hi.append(float(g.max()))
        c, n = np.unique(np.asarray(las.classification), return_counts=True)
        for k, v in zip(c.tolist(), n.tolist()):
            classes[k] = classes.get(k, 0) + v
        h = las.header
        keep = ~np.isin(np.asarray(las.classification), R.DROP)
        dens.append(keep.sum() / ((h.maxs[0] - h.mins[0]) * (h.maxs[1] - h.mins[1]) * R.FT ** 2))
        fmt = f"LAS {h.version}, point format {h.point_format.id}, CRS {h.parse_crs().name if h.parse_crs() else 'n/a'}"
    to_date = lambda s: (GPS_EPOCH + pd.to_timedelta(s + 1e9, unit="s"))  # noqa: E731  adjusted GPS time
    days = sorted({to_date(a).date() for a in lo} | {to_date(b).date() for b in hi})
    tot = sum(classes.values())
    return {"first": str(to_date(min(lo)).date()), "last": str(to_date(max(hi)).date()), "tile_days": [str(d) for d in days],
            "format": fmt, "tiles": len(t), "density_pts_m2_median": float(np.median(dens)),
            "class_share": {int(k): round(v / tot, 4) for k, v in sorted(classes.items())}, "truncated": truncated}


if __name__ == "__main__":
    if sys.argv[1] == "dates":
        info = {str(y): epoch_info(y) for y in (2018, 2022, 2025)}
        (OUT / "lidar_dates.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info, indent=2))
        sys.exit()

    info = json.loads((OUT / "lidar_dates.json").read_text())
    gmin = {y: np.load(ROOT / f"data/interim/rasters_{y}.npz")["gmin"] for y in (2018, 2022, 2025)}

    def vreg(y0, y1):
        """Ground-elevation difference y1 - y0 on 1.5 m cells with ground returns in both epochs, and per ~300 m block."""
        both = np.isfinite(gmin[y0]) & np.isfinite(gmin[y1])
        d = (gmin[y1] - gmin[y0])[both] * R.FT
        blk = 600 // R.DTM_FACTOR  # 600 cells of 0.5 m = 300 m
        rr, cc = np.nonzero(both)
        bdf = pd.DataFrame({"b": (rr // blk) * 1000 + cc // blk, "d": d}).groupby("b").d.agg(["median", "size"])
        bdf = bdf[bdf["size"] > 500]
        return (f"- {y1} − {y0}: {both.sum():,} cells; median difference **{np.median(d):+.3f} m**, IQR "
                f"{np.subtract(*np.percentile(d, [75, 25])):.3f} m. Per ~300 m block ({len(bdf)} blocks with > 500 "
                f"cells): median of block medians {bdf['median'].median():+.3f} m, range {bdf['median'].min():+.3f} "
                f"to {bdf['median'].max():+.3f} m.")
    p = P.download(T.study_area())
    cov = p.groupby(p.permitissuedate.dt.year).agg(n=("permitnumber", "size"), opa=("opa", lambda s: s.notna().mean()),
                                                   xy=("lat", lambda s: s.notna().mean()))
    rows18, rows22, rows25 = info["2018"], info["2022"], info["2025"]
    md = f"""# QC — LiDAR epochs and permits (ZIP 19146)

## LiDAR

| | 2018 | 2022 | 2025 |
|---|---|---|---|
| Source | PASDA `phillyLiDAR/LAS2018` (zipped LAS) | PASDA `phillyLiDAR/2022/LAS/Classified_LAS` | PASDA `phillyLiDAR/2025/LAS` (posted 2026-01-12/15) |
| Tiles (study area + 300 ft) | {rows18['tiles']} | {rows22['tiles']} | {rows25['tiles']} |
| Format / CRS | {rows18['format']} | {rows22['format']} | {rows25['format']}; x/y PA South US ft, z US ft (ground matches 2018) |
| **Acquisition** (GPS time) | **{rows18['first']} → {rows18['last']}** | **{rows22['first']} → {rows22['last']}** | **{rows25['first']} → {rows25['last']}** |
| Flight days in study tiles | {', '.join(rows18['tile_days'])} | {', '.join(rows22['tile_days'])} | {', '.join(rows25['tile_days'])} |
| Median density (tile, classes 7/12/18 dropped) | {rows18['density_pts_m2_median']:.1f} pts/m² | {rows22['density_pts_m2_median']:.1f} pts/m² | {rows25['density_pts_m2_median']:.1f} pts/m² |
| Class shares | {rows18['class_share']} | {rows22['class_share']} | {rows25['class_share']} |

Classes used: 2 ground (DTM); everything else except noise (7, 18) and overlap (12) for the DSM. Vegetation comes
from return structure in every epoch (multi-return pulses; masked when they are the majority in a 0.5 m cell),
because the 2022 "Classified_LAS" only separates ground. 2018 and 2025 also carry building (6) and vegetation classes.

CRS: 2018 is EPSG:2272 and 2022 is EPSG:6565 (both PA South, US ft; different NAD83 realisations, cm-level). The
2025 headers carry no CRS.

**2025 acquisition date.** PASDA's data summary (dataset 7158) and the BTAA record give the acquisition as
"April 2022" and repeat the 2022 description. The point GPS times (adjusted standard GPS time) show that every
tile in the study area was flown on the date in the table above, and the LAS headers were written in 2025 (day 196).
The 2022 metadata was evidently copied over; the point timestamps are the date used in this project.

**Truncated 2025 files on PASDA** (shorter than their headers; the server's Content-Length is the truncated size):
{'; '.join(rows25['truncated']) or 'none'}. Whole points are read; the gap shows as missing coverage.

## Vertical registration (ground, 1.5 m cells with ground returns in both epochs)

{vreg(2018, 2022)}
{vreg(2018, 2025)}

nDSM is computed against each epoch's own DTM, so a vertical offset of this size cancels out of ΔnDSM; it is about
1–2% of the 2.5 m change threshold.

## Permit location coverage (permits in the study bounding box)

{cov.round(3).to_markdown()}

Older (pre-2020, Hansen) and newer (ECLIPSE) permits both carry OPA numbers and coordinates in the current Carto table.
"""
    (OUT / "qc.md").write_text(md, encoding="utf-8")
    print(md)
