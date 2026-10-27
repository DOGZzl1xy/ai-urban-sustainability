"""Stage 7, RQ2 (results/citywide/stage7_plan.md): where, how much and when the study area densified, and how that
relates to transit access and neighborhood socioeconomic change. Local; aggregated outputs only (no addresses).

    POPUP_DATA=citywide python scripts/stage7_rq2.py

Inputs: data/interim/stage6_units.parquet (per unit: cells, LiDAR events and coverage per period, model flags per
pair), results/citywide/m4/round2_test.json (count calibration), SEPTA GTFS, ACS 5-year 2015-2019 / 2019-2023
(block groups), TIGER 2019 / 2020 block groups.
Outputs: results/citywide/stage7/rq2_*.csv, rq2_nb_models.csv, rq2_summary.json, figs/rq2_*.png
"""

import io
import json
import sys
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
from rasters import FT  # noqa: E402
from viz import PALETTE, style  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
OUT = ROOT / "results/citywide/stage7"
FIG = OUT / "figs"
PER = {"2018_2022": 4, "2022_2025": 3}
EVENTS = ("new", "popup", "demo")
CPI = 304.702 / 255.657          # CPI-U annual average 2023 / 2019: 2019 dollars -> 2023 dollars
BANDS = [0, 400, 800, 1600, np.inf]
BAND_NAMES = ["< 400 m", "400–800 m", "800–1,600 m", "> 1,600 m"]


def cell_center(key, m):
    s = m / FT
    i, j = (int(v) for v in key.split("_"))
    return (i + 0.5) * s, (j + 0.5) * s


def rail_and_bus_stops():
    """Rail-type stops (GTFS route types 0 trolley, 1 subway / elevated, 2 regional rail) and bus stops, EPSG:2272."""
    z = zipfile.ZipFile(ROOT / "data/raw/transit/gtfs_public.zip")
    rail, bus = [], []
    for name in ("google_bus.zip", "google_rail.zip"):
        f = zipfile.ZipFile(io.BytesIO(z.read(name)))
        routes = pd.read_csv(f.open("routes.txt"), usecols=["route_id", "route_type"], dtype={"route_id": str})
        trips = pd.read_csv(f.open("trips.txt"), usecols=["route_id", "trip_id"], dtype=str)
        st = pd.read_csv(f.open("stop_times.txt"), usecols=["trip_id", "stop_id"], dtype=str).drop_duplicates()
        stops = pd.read_csv(f.open("stops.txt"), usecols=["stop_id", "stop_lat", "stop_lon"], dtype={"stop_id": str})
        t = st.merge(trips, on="trip_id").merge(routes, on="route_id")[["stop_id", "route_type"]].drop_duplicates()
        s = stops.merge(t, on="stop_id")
        rail.append(s[s.route_type.isin([0, 1, 2])])
        bus.append(s[s.route_type.isin([3, 11])])
    out = []
    for d in (rail, bus):
        d = pd.concat(d).drop_duplicates(["stop_lat", "stop_lon"])
        g = gpd.GeoSeries(gpd.points_from_xy(d.stop_lon, d.stop_lat), crs=4326).to_crs(2272)
        out.append(np.c_[g.x, g.y])
    return out


def block_groups_2019():
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2019_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    bg = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2019_42_bg.zip'}!{name}")
    return bg[bg.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]]


def acs_change():
    """ACS 2015-2019 on 2019 block groups; 2019-2023 (2020 geography) interpolated onto them by overlap area:
    counts by area share, medians as overlap-area x housing-unit weighted means (an approximation)."""
    a19 = pd.read_csv(ROOT / "data/raw/acs/acs5_2019_bg_philadelphia.csv", dtype={"geoid": str})
    a23 = pd.read_csv(ROOT / "data/raw/acs/acs5_2023_bg_philadelphia.csv", dtype={"geoid": str})
    for a in (a19, a23):
        for c in a.columns[1:]:
            a[c] = pd.to_numeric(a[c], errors="coerce")
            a.loc[a[c] < 0, c] = np.nan            # Census sentinels (-666666666 etc.)
    b19 = block_groups_2019()
    with zipfile.ZipFile(ROOT / "data/raw/boundaries/tl_2020_42_bg.zip") as zf:
        name = next(n for n in zf.namelist() if n.endswith(".shp"))
    b20 = gpd.read_file(f"zip://{ROOT / 'data/raw/boundaries/tl_2020_42_bg.zip'}!{name}")
    b20 = b20[b20.COUNTYFP == "101"].to_crs(2272)[["GEOID", "geometry"]].merge(a23, left_on="GEOID", right_on="geoid")
    b20["src_area"] = b20.area
    ov = gpd.overlay(b19.rename(columns={"GEOID": "bg"}), b20, how="intersection")
    ov["w"] = ov.area / ov.src_area
    ov["wm"] = ov.area * ov.B25001.fillna(0)
    rows = []
    for bgid, g in ov.groupby("bg"):
        r = {"bg": bgid, "pop23": (g.B01003 * g.w).sum(), "hu23": (g.B25001 * g.w).sum()}
        for v, k in (("B19013", "inc23"), ("B25064", "rent23")):
            m = g[v].notna() & (g.wm > 0)
            r[k] = np.average(g.loc[m, v], weights=g.loc[m, "wm"]) if m.any() else np.nan
        rows.append(r)
    i23 = pd.DataFrame(rows)
    a = a19.rename(columns={"geoid": "bg", "B01003": "pop19", "B25001": "hu19", "B19013": "inc19",
                            "B19013_moe": "inc19_moe", "B25064": "rent19"})[["bg", "pop19", "hu19", "inc19",
                                                                              "inc19_moe", "rent19"]]
    a = a.merge(i23, on="bg", how="left")
    a["inc19_real23"] = a.inc19 * CPI
    a["inc_change"] = a.inc23 / a.inc19_real23 - 1          # real change, 2023 dollars
    a["pop_change"] = a.pop23 / a.pop19 - 1
    return a


def rates(df, groups):
    """Events per 1,000 covered units per year, per LiDAR period, by groups."""
    out = []
    for p, yrs in PER.items():
        d = df[df[f"cov{p}"]]
        g = d.groupby(groups) if groups else [("all", d)]
        for key, x in g:
            r = {"period": p.replace("_", "–"), "covered_units": len(x)}
            if groups:
                r.update(dict(zip(groups if isinstance(groups, list) else [groups],
                                  key if isinstance(key, tuple) else (key,))))
            for e in EVENTS:
                n = int(x[f"L{p}_{e}"].fillna(False).astype(bool).sum())
                r[f"{e}_events"] = n
                r[f"{e}_per_1000_per_year"] = 1000 * n / len(x) / yrs if len(x) else np.nan
            out.append(r)
    return pd.DataFrame(out)


def yearly_new(base, fac):
    """Annual new construction per 1,000 units on each anchor's common unit set (model years are estimates)."""
    rows = []
    for a, (t1, t2), lid, end, ny in ((2018, (2019, 2020), "2018_2022", 2022, 2),
                                      (2022, (2023, 2024), "2022_2025", 2025, 1)):
        com = base[base[f"cov{lid}"] & base[f"sc{a}_{t1}"] & base[f"sc{a}_{t2}"]]
        n = len(com)
        e1 = com[f"M{a}_{t1}_new"].fillna(False).astype(bool).sum() * fac
        e2 = com[f"M{a}_{t2}_new"].fillna(False).astype(bool).sum() * fac
        L = com[f"L{lid}_new"].fillna(False).astype(bool).sum()
        for year, inc, yrs, src in ((t1, e1, 1, "model"), (t2, e2 - e1, 1, "model"), (end, L - e2, ny, "LiDAR − model")):
            rows.append({"anchor": a, "year": year, "units": n, "cumulative": {t1: e1, t2: e2, end: L}[year],
                         "increment": inc, "years_in_increment": yrs, "per_1000_per_year": 1000 * inc / n / yrs,
                         "source": src})
    return pd.DataFrame(rows)


def concentration(cells, col):
    v = np.sort(cells[col].to_numpy())[::-1]
    k = max(1, int(round(0.1 * len(v))))
    return float(v[:k].sum() / v.sum()) if v.sum() else np.nan


def main():
    import statsmodels.formula.api as smf
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(ROOT / "data/interim/stage6_units.parquet")
    fac = json.load(open(ROOT / "results/citywide/m4/round2_test.json"))["models"]["m4s_s0"][
        "calibration_from_validation"]["new"]["factor"]
    main_u = base[~base.hr_c250]
    summary = {"units_in_mask": len(base), "units_main": len(main_u)}
    # --- how much, by period and area type
    r_all = rates(main_u, None)
    r_type = rates(main_u, "area_type")
    r_all.to_csv(OUT / "rq2_rates_study_area.csv", index=False)
    r_type.to_csv(OUT / "rq2_rates_by_area_type.csv", index=False)
    rates(main_u, "zip").to_csv(OUT / "rq2_rates_by_zip.csv", index=False)
    summary["rates_study_area"] = r_all.to_dict("records")
    # the 2018 flight covers fewer, more central units: compare the periods on units covered in BOTH
    same = rates(main_u[main_u.cov2018_2022 & main_u.cov2022_2025], None)
    same.to_csv(OUT / "rq2_rates_same_units.csv", index=False)
    summary["rates_units_covered_in_both_periods"] = same.to_dict("records")
    # --- when: yearly new construction
    yr = yearly_new(main_u, fac)
    yr.to_csv(OUT / "rq2_yearly_new.csv", index=False)
    # --- where: 250 m cells, concentration and maps
    cells = []
    for key, x in main_u.groupby("c250"):
        r = {"cell": key, "units": len(x)}
        for p in PER:
            c = x[f"cov{p}"]
            r[f"cov_{p}"] = int(c.sum())
            for e in ("new", "popup"):
                r[f"{e}_{p}"] = int(x.loc[c, f"L{p}_{e}"].fillna(False).astype(bool).sum())
        cells.append(r)
    cells = pd.DataFrame(cells)
    exp = cells.cov_2018_2022 * 4 + cells.cov_2022_2025 * 3
    for e in ("new", "popup"):
        cells[f"{e}_all"] = cells[f"{e}_2018_2022"] + cells[f"{e}_2022_2025"]
        cells[f"{e}_rate"] = np.where(exp >= 50 * 7, 1000 * cells[f"{e}_all"] / exp, np.nan)
    summary["concentration_top10pct_cells"] = {e: concentration(cells, f"{e}_all") for e in ("new", "popup")}
    summary["cells_250m_with_any"] = {e: float((cells[f"{e}_all"] > 0).mean()) for e in ("new", "popup")}
    cells.to_csv(OUT / "rq2_cells_250m.csv", index=False)
    s = 250 / FT
    geo = gpd.GeoDataFrame(cells, geometry=[box(int(k.split("_")[0]) * s, int(k.split("_")[1]) * s,
                                                (int(k.split("_")[0]) + 1) * s, (int(k.split("_")[1]) + 1) * s)
                                            for k in cells.cell], crs=2272)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.5))
    for ax, e, t in zip(axes, ("new", "popup"), ("New construction", "Vertical additions (pop-ups)")):
        geo.plot(ax=ax, color=PALETTE["muted"], lw=0)
        g = geo[geo[f"{e}_rate"] > 0]
        vmax = float(g[f"{e}_rate"].quantile(0.99))   # a few tiny cells would otherwise flatten the scale
        g.plot(ax=ax, column=f"{e}_rate", cmap="YlOrRd", lw=0, legend=True, vmin=0, vmax=vmax,
               legend_kwds={"label": f"per 1,000 covered units per year (scale capped at the 99th pct, {vmax:.0f})",
                            "shrink": 0.6})
        ax.set_title(f"{t}, LiDAR 2018→2025 (250 m; grey = none)", fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG / "rq2_map_250m.png", dpi=170)
    plt.close(fig)
    # --- transit access and socioeconomic change (block groups and units)
    rail, bus = rail_and_bus_stops()
    xy = np.array([cell_center(k, 100) for k in main_u.c100])
    d_rail, _ = cKDTree(rail).query(xy)
    main_u = main_u.assign(rail_m=d_rail * FT,
                           bus400=[len(v) for v in cKDTree(bus).query_ball_point(xy, 400 / FT)])
    main_u["rail_band"] = pd.cut(main_u.rail_m, BANDS, labels=BAND_NAMES, right=False)
    rb = rates(main_u, "rail_band")
    rb.to_csv(OUT / "rq2_rates_by_rail_distance.csv", index=False)
    acs = acs_change()
    bgu = main_u[~main_u.hr_bg & main_u.bg.notna()]
    agg = {"units": ("unit", "size"), "rail_m": ("rail_m", "mean"), "bus400": ("bus400", "mean"),
           "rowhouse": ("area_type", lambda x: float((x == "rowhouse").mean()))}
    for p in PER:
        agg[f"cov_{p}"] = (f"cov{p}", "sum")
    bgd = bgu.groupby("bg").agg(**agg).reset_index()
    for p in PER:
        for e in ("new", "popup"):
            m = bgu[bgu[f"cov{p}"]].groupby("bg")[f"L{p}_{e}"].apply(lambda x: x.fillna(False).astype(bool).sum())
            bgd = bgd.merge(m.rename(f"{e}_{p}").reset_index(), on="bg", how="left")
    bgd = bgd.fillna({c: 0 for c in bgd.columns if c.startswith(("new_", "popup_"))})
    bgd["exposure"] = bgd.cov_2018_2022 * 4 + bgd.cov_2022_2025 * 3
    for e in ("new", "popup"):
        bgd[e] = bgd[f"{e}_2018_2022"] + bgd[f"{e}_2022_2025"]
    bgd = bgd.merge(acs, on="bg", how="left")
    bgd["rail_band"] = pd.cut(bgd.rail_m, BANDS, labels=BAND_NAMES, right=False)
    ok = (bgd.exposure >= 50 * 7) & bgd.inc19.notna() & bgd.inc_change.notna() & bgd.pop_change.notna()
    m = bgd[ok].copy()
    m["inc_q"] = pd.qcut(m.inc19, 5, labels=[f"Q{i}" for i in range(1, 6)])
    m["incchg_q"] = pd.qcut(m.inc_change, 5, labels=[f"Q{i}" for i in range(1, 6)])
    m["inc_change_z"] = (m.inc_change - m.inc_change.mean()) / m.inc_change.std()
    m["pop_change_z"] = (m.pop_change.clip(-0.5, 1) - m.pop_change.clip(-0.5, 1).mean()) / m.pop_change.clip(-0.5, 1).std()
    m.to_csv(OUT / "rq2_block_groups.csv", index=False)
    by_q = []
    for qcol, name in (("inc_q", "baseline income 2015–19"), ("incchg_q", "real income change")):
        for q, x in m.groupby(qcol, observed=True):
            by_q.append({"by": name, "quintile": q, "block_groups": len(x),
                         "median_value": float((x.inc19 if qcol == "inc_q" else x.inc_change).median()),
                         **{f"{e}_per_1000_per_year": 1000 * x[e].sum() / x.exposure.sum() for e in ("new", "popup")}})
    pd.DataFrame(by_q).to_csv(OUT / "rq2_rates_by_income.csv", index=False)
    nb = []
    for e in ("new", "popup"):
        f = (f"{e} ~ C(rail_band, Treatment('< 400 m')) + C(inc_q, Treatment('Q3')) + inc_change_z + pop_change_z "
             f"+ rowhouse")
        fit = smf.negativebinomial(f, data=m, exposure=m.exposure).fit(disp=0, maxiter=200)
        ci = fit.conf_int()
        for v in fit.params.index:
            if v in ("Intercept", "alpha"):
                continue
            nb.append({"outcome": e, "term": v, "irr": float(np.exp(fit.params[v])),
                       "ci_low": float(np.exp(ci.loc[v, 0])), "ci_high": float(np.exp(ci.loc[v, 1])),
                       "p": float(fit.pvalues[v]), "n_block_groups": int(fit.nobs), "alpha": float(fit.params["alpha"])})
    pd.DataFrame(nb).to_csv(OUT / "rq2_nb_models.csv", index=False)
    summary["block_groups_in_models"] = int(len(m))
    # --- figures: yearly series and rates by rail distance / income
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()
    ax = axes[0]
    lab = {2019: "2019*", 2020: "2020*", 2022: "2021–22\n(avg)", 2023: "2023*", 2024: "2024*", 2025: "2025"}
    for a, c in ((2018, PALETTE["primary"]), (2022, PALETTE["accent"])):
        d = yr[yr.anchor == a]
        ax.bar([lab[y] for y in d.year], d.per_1000_per_year, color=c, alpha=0.85,
               label=f"{a} anchor ({int(d.units.iloc[0]):,} units)")
    ax.set(title="New construction per year (* model estimate; others LiDAR minus model)",
           ylabel="per 1,000 units per year")
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    for ax, df_, col, t in ((axes[1], rb, "rail_band", "By distance to rail-type stop (LiDAR)"),):
        piv = df_.pivot(index=col, columns="period", values="new_per_1000_per_year").reindex(BAND_NAMES)
        piv2 = df_.pivot(index=col, columns="period", values="popup_per_1000_per_year").reindex(BAND_NAMES)
        x = np.arange(len(BAND_NAMES))
        for k, (p, c) in enumerate(zip(piv.columns, (PALETTE["primary"], PALETTE["positive"]))):
            ax.bar(x + k * 0.2 - 0.3, piv[p], 0.2, color=c, label=f"new {p}")
            ax.bar(x + k * 0.2 + 0.1, piv2[p], 0.2, color=c, alpha=0.45, label=f"pop-up {p}")
        ax.set_xticks(x, BAND_NAMES, fontsize=8)
        ax.set(title=t, ylabel="per 1,000 covered units per year")
        ax.legend(fontsize=7)
    q = pd.DataFrame(by_q)
    for ax, e, t in ((axes[2], "new", "New construction"), (axes[3], "popup", "Pop-ups")):
        for k, (name, c) in enumerate((("baseline income 2015–19", PALETTE["primary"]),
                                       ("real income change", PALETTE["accent"]))):
            d = q[q.by == name]
            ax.bar(np.arange(5) + (k - 0.5) * 0.38, d[f"{e}_per_1000_per_year"], 0.38, color=c,
                   label=f"quintile of {name}")
        ax.set_xticks(np.arange(5), ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
        ax.set(title=f"{t} by block-group income quintile (LiDAR 2018→2025)",
               ylabel="per 1,000 covered units per year")
        ax.title.set_fontsize(9)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rq2_when_transit_income.png", dpi=170)
    plt.close(fig)
    (OUT / "rq2_summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=float))
    print(yr.round(2).to_string(index=False))
    print(rb[["period", "rail_band", "covered_units", "new_per_1000_per_year", "popup_per_1000_per_year"]].round(2)
          .to_string(index=False))
    print(pd.DataFrame(by_q).round(3).to_string(index=False))
    print(pd.DataFrame(nb).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
