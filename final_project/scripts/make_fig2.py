"""Proposal figure 2 (revised proposal, 2026-10-28): LiDAR new construction and pop-ups per 250 m cell, 2018-2025.

    python scripts/make_fig2.py       -> proposal/fig2.png

Aggregated cells only (results/citywide/stage7/rq2_cells_250m.csv); no building is shown.
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rasters import FT  # noqa: E402
from viz import PALETTE, style  # noqa: E402

style()
c = pd.read_csv(ROOT / "results/citywide/stage7/rq2_cells_250m.csv")
s = 250 / FT
c = gpd.GeoDataFrame(c, crs=2272, geometry=[box(int(k.split("_")[0]) * s, int(k.split("_")[1]) * s,
                                                (int(k.split("_")[0]) + 1) * s, (int(k.split("_")[1]) + 1) * s)
                                            for k in c.cell])
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))
for ax, e, t in zip(axes, ("new", "popup"), ("(a) New construction", "(b) Vertical additions (pop-ups)")):
    c.plot(ax=ax, color=PALETTE["muted"], lw=0)
    g = c[c[f"{e}_rate"] > 0]
    vmax = float(g[f"{e}_rate"].quantile(0.99))
    g.plot(ax=ax, column=f"{e}_rate", cmap="YlOrRd", lw=0, vmin=0, vmax=vmax, legend=True,
           legend_kwds={"shrink": 0.62, "label": "per 1,000 units per year", "pad": 0.01})
    ax.set_title(t, fontsize=9, loc="left")
    ax.set_axis_off()
fig.tight_layout(pad=0.2, w_pad=0.4)
fig.savefig(ROOT / "proposal/fig2.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("proposal/fig2.png")
