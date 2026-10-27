"""Proposal Figure 1: (a) 2018→2022 change in ZIP 19146 on a 100 m grid (pop-ups and new construction),
(b) one permitted pop-up: aerial 2018 / 2022 and the LiDAR height profile along the lot. No address is shown.

Writes proposal/fig1.png.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import ortho_chips as O  # noqa: E402
import tiles as T  # noqa: E402
from rasters import CELL, FT  # noqa: E402
from viz import PALETTE, style  # noqa: E402

style()
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11})
units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
f = C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), C.Rules())
u = units[["unit", "geometry"]].merge(f[["unit", "chg_new", "chg_popup"]], on="unit")

# ---- (a) 100 m grid ----
cell = 100 / FT
cent = u.geometry.representative_point()
u["gx"], u["gy"] = (cent.x // cell).astype(int), (cent.y // cell).astype(int)
agg = u.groupby(["gx", "gy"]).agg(popup=("chg_popup", "sum"), new=("chg_new", "sum")).reset_index()
zip_poly = T.study_area(buffer_ft=0)

# ---- (b) case: a textbook pop-up among permitted, detected upward additions: full lot built in 2018, footprint
# area unchanged (< 10%), two-story start (6-9.5 m), one-story rise (2.5-4.5 m); largest share of rising cells ----
rs = pd.read_parquet(ROOT / "data/interim/recall_set_initial.parquet")
case = rs[(rs.kind == "upward") & (rs.set == "primary") & rs.detected_type.fillna(False)].drop_duplicates("unit")
case = case[(case.cover18 > 0.6) & ((case.bldg_m2_22 - case.bldg_m2_18).abs() / case.bldg_m2_18 < 0.1)
            & case.h50_18.between(6, 9.5) & (case.h50_22 - case.h50_18).between(2.5, 4.5)]
case = case.sort_values("frac_up", ascending=False).iloc[0]
g = units.set_index("unit").loc[case.unit, "geometry"]
a18, grid = C.load(2018)
a22, _ = C.load(2022)

# profile along the lot's long axis (front to back), through the centroid
rect = g.minimum_rotated_rectangle
xs, ys = rect.exterior.coords.xy
edges = [((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(4)]
(p, q) = max(edges, key=lambda e: np.hypot(e[1][0] - e[0][0], e[1][1] - e[0][1]))
d = np.array(q) - np.array(p)
d /= np.linalg.norm(d)
c = np.array(g.centroid.coords[0])
ext = 0.5 * max(np.hypot(*(np.array(e[1]) - np.array(e[0]))) for e in edges) + 8 / FT  # lot + 8 m each side
s = np.linspace(-ext, ext, 400)
px, py = c[0] + s * d[0], c[1] + s * d[1]
r, k = grid.index(px, py)
h18, h22 = a18["ndsm"][r, k], a22["ndsm"][r, k]
inside = np.array([g.buffer(0.1).contains(gpd.points_from_xy([x], [y])[0]) for x, y in zip(px, py)])

fig = plt.figure(figsize=(7.2, 3.1))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 0.55, 0.55], height_ratios=[1, 0.9], wspace=0.15, hspace=0.42)
ax = fig.add_subplot(gs[:, 0])
zip_poly.boundary.plot(ax=ax, color="0.55", lw=0.7)
off = cell * 0.18
for col, color, dx, label in [("popup", PALETTE["primary"], -off, "pop-up"), ("new", PALETTE["accent"], off, "new building")]:
    a = agg[agg[col] > 0]
    ax.scatter((a.gx + 0.5) * cell + dx, (a.gy + 0.5) * cell, s=a[col] * 5, color=color, alpha=0.75, lw=0,
               label=label)
ax.legend(frameon=False, fontsize=9.5, loc="lower left", bbox_to_anchor=(0, -0.02), handletextpad=0.2, ncol=2,
          title="dot size: 1–10 per 100 m cell", title_fontsize=8.5, alignment="left")
y0, y1 = ax.get_ylim()
ax.set_ylim(y0 - 0.35 * (y1 - y0), y1)  # room for the legend below the ZIP outline
ax.set_aspect("equal")
ax.set_axis_off()
ax.set_title("(a) ZIP 19146, 2018→2022", fontsize=11, loc="left")

line = gpd.GeoSeries([LineString(zip(px, py))], crs=2272)
for i, yr in enumerate((2018, 2022)):
    axi = fig.add_subplot(gs[0, 1 + i])
    img, (bx0, by0, bx1, by1) = O.chip_georef(g, yr, pad_m=8, out_res_m=0.1)
    axi.imshow(img, extent=(bx0, bx1, by0, by1))
    gpd.GeoSeries([g], crs=2272).to_crs(3857).boundary.plot(ax=axi, color="#ffd400", lw=1)
    line.to_crs(3857).plot(ax=axi, color="white", lw=0.8, ls="--")
    axi.set_axis_off()
    axi.set_title(f"(b) {yr}" if i == 0 else f"{yr}", fontsize=11, loc="left")
axp = fig.add_subplot(gs[1, 1:])
dist = (s - s[0]) * FT
axp.fill_between(dist, 0, np.where(inside, 40, 0), color="0.93", lw=0, step="mid")
axp.plot(dist, h18, color=PALETTE["neutral"], lw=1.2, label="LiDAR 2018")
axp.plot(dist, h22, color=PALETTE["accent"], lw=1.2, label="LiDAR 2022")
top = np.nanmax(np.r_[h18, h22]) + 2
axp.set(xlim=(dist[0], dist[-1]), ylim=(0, top), xlabel="distance along the lot (m)",
        ylabel="height (m)")
axp.legend(frameon=False, fontsize=9.5, loc="lower right", bbox_to_anchor=(1, 0.1))
axp.text(dist[inside][0] + 0.5, top - 1.5, "lot", fontsize=9.5, color="0.45", va="top")
fig.savefig(ROOT / "proposal/fig1.png", dpi=300, bbox_inches="tight")
print("case unit", case.unit, "frac_up", round(case.frac_up, 2), "median rise in lot",
      round(float(np.nanmedian((h22 - h18)[inside])), 1), "m")
