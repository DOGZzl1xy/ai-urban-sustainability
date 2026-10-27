"""Step 2: final rules -> flagged parcels, unpermitted candidates, review samples, review.html, figures.

Usage: python finalize.py RULES_TAG     (RULES_TAG = "initial" or the stem of a rules json used with detect.py)
Samples are drawn only here, after the thresholds are fixed.
"""

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
import change as C  # noqa: E402
import match as M  # noqa: E402
import ortho_chips as O  # noqa: E402
import permits as P  # noqa: E402
import tiles as T  # noqa: E402
from rasters import CELL  # noqa: E402
from viz import PALETTE, style  # noqa: E402

style()
OUT = ROOT / "results/popup_feasibility"
FIG = OUT / "figs"
TAG = sys.argv[1] if len(sys.argv) > 1 else "initial"
rules = C.Rules(**json.loads((OUT / f"{TAG}.json").read_text())) if TAG != "initial" else C.Rules()
DATES = json.loads((OUT / "lidar_dates.json").read_text())
D18, D22 = pd.Timestamp(DATES["2018"]["first"]), pd.Timestamp(DATES["2022"]["first"])

units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
f = C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), rules)
p = M.match_permits(P.classify(P.download(T.study_area())), units)
p = p[p.unit.notna()].copy()
p["unit"] = p.unit.astype(int)
nb = M.neighbours(units)
cand = M.unpermitted(f, p, nb, D18, D22)
f = f.merge(cand[["unit", "permit_on_unit", "permit_on_neighbour", "candidate"]], on="unit", how="left")
f["candidate"] = f.candidate.fillna(False).astype(bool)
f["flagged"] = f.chg_any_build | f.chg_demo
keep = ["unit", "n_cells", "cover18", "cover22", "bldg_m2_18", "bldg_m2_22", "h50_18", "h50_22", "frac_up", "frac_down",
        "dh_p50", "new_outside_fp_m2", "chg_new", "chg_demo", "chg_popup", "chg_rear", "chg_any_build", "flagged",
        "permit_on_unit", "permit_on_neighbour", "candidate"]
f[keep].to_parquet(OUT / "flagged_parcels.parquet")  # unit ids only; addresses stay in data/interim
print({c: int(f[c].sum()) for c in ["chg_new", "chg_demo", "chg_popup", "chg_rear", "flagged", "candidate"]})

# ---- samples (fixed seed, drawn after the rules are final) ----
rng = np.random.default_rng(20221)
prec = f[f.flagged].sample(min(50, f.flagged.sum()), random_state=20221).assign(role="precision")
unp = f[f.candidate].sample(min(50, f.candidate.sum()), random_state=20222).assign(role="unpermitted")
prec["item"] = [f"P{i + 1:02d}" for i in range(len(prec))]
unp["item"] = [f"U{i + 1:02d}" for i in range(len(unp))]
items = pd.concat([prec, unp]).sample(frac=1, random_state=7).reset_index(drop=True)
items = items.merge(units[["unit", "address", "opas", "geometry"]], on="unit")
items[["item", "role", "unit"]].to_csv(OUT / "review_items.csv", index=False)

# ---- review page ----
import review_page as RP  # noqa: E402

html = RP.build(items, "Change review — ZIP 19146", "change-review-19146-v2", f"Rules: {rules}.")

(OUT / "review.html").write_text(html, encoding="utf-8")
print(f"review.html: {len(items)} cases ({(items.role == 'precision').sum()} precision, {(items.role == 'unpermitted').sum()} unpermitted)")

# ---- figures ----
rec = pd.read_csv(OUT / f"recall_{TAG}.csv")
r = rec[(rec.set == "primary")]
fig, ax = plt.subplots(figsize=(6, 3.6))
kinds = ["new", "upward", "rear"]
for i, meas in enumerate(["detected_any", "detected_type"]):
    rr = r[r.measure == meas].set_index("kind").reindex(kinds)
    x = np.arange(3) + (i - 0.5) * 0.36
    ax.bar(x, rr.recall, 0.34, yerr=[rr.recall - rr.lo, rr.hi - rr.recall], capsize=3,
           color=[PALETTE["primary"], PALETTE["accent"]][i], label={"detected_any": "any change detected",
                                                                     "detected_type": "matching type detected"}[meas])
    for xi, (n, v) in zip(x, zip(rr.n, rr.recall)):
        ax.text(xi, 0.02, f"n={n}", ha="center", fontsize=7, color="white")
ax.axhline(0.7, color="0.4", ls="--", lw=0.8)
ax.set_xticks(range(3), ["new construction", "upward addition", "rear addition (not judged)"])
ax.set(ylim=(0, 1.05), ylabel="Recall", title="Recall on permitted work completed between LiDAR epochs (95% CI)")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "recall.png", dpi=170)

# aggregated map: change types on a 100 m grid (no individual parcels shown)
u = units.merge(f, on="unit")
cent = u.geometry.representative_point()
cell = 328.0  # 100 m in ft
u["gx"], u["gy"] = (cent.x // cell).astype(int), (cent.y // cell).astype(int)
agg = u.groupby(["gx", "gy"]).agg(n=("unit", "size"), popup=("chg_popup", "sum"), new=("chg_new", "sum"),
                                  rear=("chg_rear", "sum"), cand=("candidate", "sum")).reset_index()
from shapely.geometry import box  # noqa: E402
agg = gpd.GeoDataFrame(agg, geometry=[box(x * cell, y * cell, (x + 1) * cell, (y + 1) * cell) for x, y in zip(agg.gx, agg.gy)], crs=2272)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
z = T.study_area(buffer_ft=0)
for ax, col, title in zip(axes, ["popup", "new", "cand"], ["pop-up / upward additions", "new construction",
                                                          "flagged change without any permit (candidates)"]):
    z.boundary.plot(ax=ax, color="0.5", lw=0.6)
    agg[agg[col] > 0].plot(column=col, ax=ax, cmap="magma_r", legend=True, legend_kwds={"shrink": 0.6})
    ax.set_title(f"{title}\nper 100 m cell, 2018→2022", fontsize=9)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(FIG / "change_map.png", dpi=170)

# three typical pop-ups: permitted upward additions detected with the largest share of rising cells
rs = pd.read_parquet(ROOT / f"data/interim/recall_set_{TAG}.parquet")
ex = rs[(rs.kind == "upward") & (rs.set == "primary") & rs.detected_type.fillna(False)].sort_values("frac_up", ascending=False)
for i, (_, r) in enumerate(ex.head(3).iterrows(), 1):
    g = units.set_index("unit").loc[r.unit, "geometry"]
    aerial = O.strip(g, years=(2018, 2022), size=420)
    dh = RP.dh_chip(g, size=420)
    img = Image.new("RGB", (aerial.width + 8 + dh.width, 420), (255, 255, 255))
    img.paste(aerial, (0, 0))
    img.paste(dh, (aerial.width + 8, 0))
    img.save(FIG / f"popup_case_{i}.png")
print("figures written")
