"""LiDAR rule spot-check in new areas (results/citywide/spotcheck_plan.md).

    python scripts/spotcheck.py          # draw once -> spotcheck_items.csv; build review_spotcheck.html (local only)
    python scripts/spotcheck.py score    # review_spotcheck_labels.csv -> spotcheck_scores.json
"""

import io
import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match import bootstrap_rate  # noqa: E402
from rasters import CELL, Grid  # noqa: E402

OUT = ROOT / "results/citywide"
CHG = ROOT / "data/interim/lidar_change_city"
RAS = ROOT / "data/interim/citywide_rasters"
SEED = 20261026
PER_TYPE, PER_PERIOD = 20, 10
PERIODS = ((2018, 2022), (2022, 2025))
STAGE3 = {"19121", "19122", "19123", "19125", "19130", "19132", "19143", "19146", "19147"}


def pool():
    zt = pd.read_csv(OUT / "splits_citywide_zips.csv", dtype={"zip": str})
    zips = [z for z in zt[~zt.test].zip if z not in STAGE3]
    rows = []
    for z in zips:
        for y0, y1 in PERIODS:
            d = pd.read_parquet(CHG / f"{z}_{y0}_{y1}.parquet")
            cov = (d.data_18 > 0.95) & (d.data_22 > 0.95) & d.cover18.notna() & d.cover22.notna()
            d = d[d.in_mask & cov & (d.chg_new | d.chg_popup)]
            rows.append(d.assign(zip=z, y0=y0, y1=y1))
    return pd.concat(rows, ignore_index=True)


def draw():
    path = OUT / "spotcheck_items.csv"
    if path.exists():   # drawn once; never redraw over labels
        return pd.read_csv(path, dtype={"zip": str})
    p = pool()
    rng = np.random.default_rng(SEED)
    picks = []
    for t in ("rowhouse", "detached", "mixed"):
        g = p[p.area_type == t]
        by = {per: g[(g.y0 == per[0]) & (g.y1 == per[1])] for per in PERIODS}
        take = {per: min(PER_PERIOD, len(by[per])) for per in PERIODS}
        for per in PERIODS:   # fill up from the other period if one is short
            other = [q for q in PERIODS if q != per][0]
            take[per] = min(len(by[per]), take[per] + max(0, PER_PERIOD - take[other]))
        for per in PERIODS:
            d = by[per]
            idx = rng.choice(len(d), size=take[per], replace=False)
            picks.append(d.iloc[np.sort(idx)])
        print(f"{t}: pool {len(g)} ({', '.join(f'{a}-{b}: {len(by[(a, b)])}' for a, b in PERIODS)}); "
              f"drew {sum(take.values())}")
    s = pd.concat(picks, ignore_index=True)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)   # mixed order on the page
    s["item"] = [f"S{i + 1:02d}" for i in range(len(s))]
    s["role"] = "spotcheck"
    s[["item", "role", "zip", "unit", "y0", "y1", "area_type"]].to_csv(path, index=False)
    return s[["item", "role", "zip", "unit", "y0", "y1", "area_type"]]


def dh_chips(rows, pad_m=10, size=300):
    """Height-change chip per row from the citywide rasters, one ZIP and period at a time (rasters are large)."""
    out = {}
    for (z, y0, y1), g in rows.groupby(["zip", "y0", "y1"]):
        a, b = np.load(RAS / f"{z}_{y0}.npz"), np.load(RAS / f"{z}_{y1}.npz")
        grid = Grid(float(a["x0"]), float(a["y1"]), int(a["nx"]), int(a["ny"]))
        na, nb = a["ndsm"], b["ndsm"]
        for _, r in g.iterrows():
            x0, yy0, x1, yy1 = r.geometry.bounds
            cx, cy = (x0 + x1) / 2, (yy0 + yy1) / 2
            half = max(x1 - x0, yy1 - yy0) / 2 + pad_m / 0.3048
            (ra,), (ca,) = grid.index(np.array([cx - half]), np.array([cy + half]))
            (rb,), (cb,) = grid.index(np.array([cx + half]), np.array([cy - half]))
            ra, ca = max(ra, 0), max(ca, 0)
            sub = nb[ra:rb, ca:cb] - na[ra:rb, ca:cb]
            ext = (grid.x0 + ca * CELL, grid.x0 + cb * CELL, grid.y1 - rb * CELL,
                   grid.y1 - ra * CELL)
            fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
            ax.imshow(np.clip(sub, -6, 6), cmap="RdBu_r", vmin=-6, vmax=6, extent=ext)
            geom = r.geometry if r.geometry.geom_type == "Polygon" else max(r.geometry.geoms, key=lambda q: q.area)
            ax.plot(*geom.exterior.xy, color="k", lw=1)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            out[r["item"]] = Image.open(buf).convert("RGB").resize((size, size))
        del a, b, na, nb
    return out


def page(items):
    import review_page as RP

    feats = pd.concat([pd.read_parquet(CHG / f"{z}_{y0}_{y1}.parquet").assign(zip=z, y0=y0, y1=y1)
                       for z, y0, y1 in items[["zip", "y0", "y1"]].drop_duplicates().itertuples(index=False)])
    units = pd.concat([gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")[
        ["unit", "address", "opas", "geometry"]].assign(zip=z) for z in items.zip.unique()])
    rows = (items.merge(feats.drop(columns="area_type"), on=["zip", "unit", "y0", "y1"])
            .merge(units, on=["zip", "unit"]))
    rows = gpd.GeoDataFrame(rows, geometry="geometry", crs=2272).set_index("item", drop=False).loc[items.item]
    chips = dh_chips(rows)
    html = RP.build(rows.reset_index(drop=True), "LiDAR rule spot-check — new areas", "spotcheck-citywide-v1",
                    "Random detections (new construction or pop-up) from ZIPs where the rules were never checked: "
                    "20 per area type (rowhouse, detached, mixed), half 2018→2022 and half 2022→2025. "
                    "Each card shows its own two years.",
                    rear_col="chg_rear", export_name="review_spotcheck_labels.csv",
                    dh=lambda r: chips[r["item"]], role_names={"spotcheck": "spot-check"})
    (OUT / "review_spotcheck.html").write_text(html, encoding="utf-8")
    print(f"review_spotcheck.html: {len(rows)} cases")


def score():
    items = pd.read_csv(OUT / "spotcheck_items.csv", dtype={"zip": str})
    lab = pd.read_csv(OUT / "review_spotcheck_labels.csv").fillna("")
    feats = pd.concat([pd.read_parquet(CHG / f"{z}_{y0}_{y1}.parquet")[["unit", "chg_new", "chg_popup"]]
                       .assign(zip=z, y0=y0, y1=y1)
                       for z, y0, y1 in items[["zip", "y0", "y1"]].drop_duplicates().itertuples(index=False)])
    m = items.merge(lab, on="item").merge(feats, on=["zip", "unit", "y0", "y1"])
    j = m[m.change.isin(["real", "none"])]

    def prec(d):
        x = (d.change == "real").to_numpy(float)
        return {"n": int(len(d)), "real": int(x.sum()), "precision": float(x.mean()) if len(d) else None,
                "ci95": bootstrap_rate(x) if len(d) else None}

    out = {"n_items": int(len(m)), "n_unsure": int((m.change == "unsure").sum()), "overall": prec(j),
           "by_area_type": {t: prec(g) for t, g in j.groupby("area_type")},
           "by_class": {"new": prec(j[j.chg_new]), "popup": prec(j[j.chg_popup])},
           "by_period": {f"{a}-{b}": prec(g) for (a, b), g in j.groupby(["y0", "y1"])},
           "false_positive_notes": j[j.change == "none"].change_note.tolist()}
    p = out["overall"]["precision"]
    out["decision"] = ("rules used citywide unchanged" if p is not None and p >= 0.85 else
                       "M4 waits: categorise false positives with the user")
    out["area_types_below_75"] = [t for t, v in out["by_area_type"].items()
                                  if v["precision"] is not None and v["precision"] < 0.75]
    (OUT / "spotcheck_scores.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        score()
    else:
        page(draw())
