"""Stage 7, RQ3 (results/citywide/stage7_plan.md): permits vs visible construction, time to detection, change without a
matching permit, and a rough floor-area / housing estimate. Local; aggregated outputs only (no addresses, no parcel ids).

    POPUP_DATA=citywide python scripts/stage7_rq3.py

Permits (data/raw/li/permits_studyarea_2014_2025.csv) are classified with src/permits.py and matched to units by OPA
account (data/interim/citywide_units opas), else by point-in-polygon. Construction comes from
data/interim/stage6_units.parquet (LiDAR events per period on covered units; model-dated new construction).
Outputs: results/citywide/stage7/rq3_*.csv, rq3_summary.json, figs/rq3_time_to_detection.png
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
import permits as PM  # noqa: E402
from rasters import FT  # noqa: E402
from viz import PALETTE, style  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
OUT = ROOT / "results/citywide/stage7"
FIG = OUT / "figs"
PERIODS = {"2018_2022": ("2018-04-01", "2022-04-01"), "2022_2025": ("2022-04-01", "2025-04-18")}  # spring flights
STOREY_M = 3.0


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_permits(zips):
    p = pd.read_csv(ROOT / "data/raw/li/permits_studyarea_2014_2025.csv",
                    dtype={"opa_account_num": str, "parcel_id_num": str, "zip": str})
    for c in ("permitissuedate", "permitcompleteddate", "certificateofoccupancydate"):
        p[c] = pd.to_datetime(p[c], errors="coerce", utc=True).dt.tz_convert(None)
    p["opa"] = p.opa_account_num.str.replace(r"\.0$", "", regex=True).str.zfill(9).where(p.opa_account_num.notna())
    p = PM.classify(p).reset_index(drop=True).rename(columns={"zip": "permit_zip"})
    p["pid"] = np.arange(len(p))
    # units: OPA -> unit, then point-in-polygon for the rest
    units = pd.concat([gpd.read_parquet(ROOT / f"data/interim/citywide_units/{z}.parquet")[
        ["unit", "opas", "geometry", "in_mask"]].assign(zip=z) for z in zips])
    units = gpd.GeoDataFrame(units[units.in_mask], geometry="geometry", crs=2272)
    ex = units[["zip", "unit", "opas"]].explode("opas").dropna(subset=["opas"])
    ex["opa"] = ex.opas.astype(str).str.zfill(9)
    m1 = p[["pid", "opa"]].dropna().merge(ex[["opa", "zip", "unit"]], on="opa")
    rest = p[~p.pid.isin(m1.pid) & p.lat.notna()]
    pts = gpd.GeoDataFrame(rest[["pid"]], geometry=gpd.points_from_xy(rest.lng, rest.lat), crs=4326).to_crs(2272)
    m2 = gpd.sjoin(pts, units[["zip", "unit", "geometry"]], predicate="within")[["pid", "zip", "unit"]]
    link = pd.concat([m1[["pid", "zip", "unit"]].assign(how="opa"), m2.assign(how="point")]).drop_duplicates(
        ["pid", "zip", "unit"])
    return p, link, units


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    zips = sorted(zt.zip)
    base = pd.read_parquet(ROOT / "data/interim/stage6_units.parquet")
    base = base[~base.hr_c250]
    p, link, units = load_permits(zips)
    pu = link.merge(p, on="pid").merge(base, on=["zip", "unit"])
    S = {"permits": int(len(p)), "permits_linked_to_main_units": int(pu.pid.nunique()),
         "link_by_opa_share": float((link.how == "opa").mean())}
    # ---- A. permit -> visible construction (one row per unit: earliest permit of the kind in the window)
    rows = []
    for kind, flag, ev in (("new construction", "is_new", "new"), ("addition (any)", "is_addition", "popup"),
                           ("upward addition", "is_upward", "popup")):
        w = pu[pu[flag] & (pu.permitissuedate >= "2018-01-01") & (pu.permitissuedate < "2021-01-01")]
        w = w.sort_values("permitissuedate").drop_duplicates(["zip", "unit"])
        c22 = w[w.cov2018_2022]
        by22 = c22[f"L2018_2022_{ev}"].fillna(False).astype(bool)
        c25 = w[w.cov2018_2022 & w.cov2022_2025]
        by25 = (c25[f"L2018_2022_{ev}"].fillna(False).astype(bool) | c25[f"L2022_2025_{ev}"].fillna(False).astype(bool))
        anyb = (c25[f"L2018_2022_grew"].fillna(False).astype(bool) | c25[f"L2022_2025_grew"].fillna(False).astype(bool))
        for label, n, k in (("by 2022 (LiDAR)", len(c22), int(by22.sum())), ("by 2025 (LiDAR)", len(c25), int(by25.sum())),
                            ("any growth by 2025", len(c25), int(anyb.sum()))):
            lo, hi = wilson(k, n)
            rows.append({"permit": kind, "issued": "2018-2020", "outcome": f"{ev if 'growth' not in label else 'grew'} "
                         f"{label}", "units": n, "detected": k, "share": k / n if n else np.nan, "ci_low": lo, "ci_high": hi})
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "rq3_permit_to_construction.csv", index=False)
    # ---- B. time from permit issue to first detection (new construction; units tracked in all six years)
    t = base[base.cov2018_2022 & base.cov2022_2025 & base.sc2018_2019 & base.sc2018_2020 & base.sc2022_2023
             & base.sc2022_2024].copy()
    f = lambda c: t[c].fillna(False).astype(bool)  # noqa: E731
    built = {2019: f("M2018_2019_new")}
    built[2020] = built[2019] | f("M2018_2020_new")
    built[2022] = built[2020] | f("L2018_2022_new")
    built[2023] = built[2022] | f("M2022_2023_new")
    built[2024] = built[2023] | f("M2022_2024_new")
    built[2025] = built[2024] | f("L2022_2025_new")
    first = pd.Series(np.nan, index=t.index)
    for y in (2025, 2024, 2023, 2022, 2020, 2019):
        first[built[y]] = y
    t["first_year"] = first
    nw = pu[pu.is_new & (pu.permitissuedate >= "2018-04-01") & (pu.permitissuedate < "2022-01-01")]
    nw = nw.sort_values("permitissuedate").drop_duplicates(["zip", "unit"])
    nw = nw.merge(t[["zip", "unit", "first_year"]], on=["zip", "unit"])
    nw["issue_year"] = nw.permitissuedate.dt.year
    nw["lag"] = nw.first_year - nw.issue_year
    tt = []
    for iy, g in nw.groupby("issue_year"):
        d = {"issue_year": int(iy), "units": len(g), "never_detected_by_2025": float(g.first_year.isna().mean()),
             "detected_before_issue_year": float((g.lag < 0).mean())}
        for k in range(0, 7):
            d[f"detected_within_{k}y"] = float((g.lag.between(0, k)).mean())
        d["median_lag_years_detected"] = float(g.lag[g.lag >= 0].median()) if (g.lag >= 0).any() else np.nan
        tt.append(d)
    tt = pd.DataFrame(tt)
    tt.to_csv(OUT / "rq3_time_to_detection.csv", index=False)
    # ---- C. change without a matching permit (related permit on the unit; sensitivity: within 10 m of the unit)
    rel = pu[pu.is_related][["zip", "unit", "permitissuedate"]]
    relp = p[p.is_related & p.lat.notna()]
    relp = gpd.GeoDataFrame(relp[["pid", "permitissuedate"]], geometry=gpd.points_from_xy(relp.lng, relp.lat),
                            crs=4326).to_crs(2272)
    ug = units.set_index(["zip", "unit"]).geometry
    uq = []
    for per, (s, e) in PERIODS.items():
        lo = (pd.Timestamp(s) - pd.DateOffset(years=3))
        r_ = rel[(rel.permitissuedate >= lo) & (rel.permitissuedate < e)]
        has = set(zip(r_.zip, r_.unit))
        rp = relp[(relp.permitissuedate >= lo) & (relp.permitissuedate < e)]
        for ev in ("new", "popup"):
            evu = base[base[f"cov{per}"] & base[f"L{per}_{ev}"].fillna(False).astype(bool)][["zip", "unit"]]
            on_unit = np.array([(z, u) in has for z, u in zip(evu.zip, evu.unit)])
            buf = gpd.GeoDataFrame(evu.reset_index(drop=True), crs=2272,
                                   geometry=ug.loc[list(zip(evu.zip, evu.unit))].buffer(10 / FT).to_numpy())
            near = gpd.sjoin(buf, rp[["geometry"]], predicate="intersects").index.unique()
            within10 = np.zeros(len(evu), bool)
            within10[near] = True
            for label, ok in (("related permit on the unit", on_unit), ("related permit within 10 m", within10 | on_unit)):
                k, n = int((~ok).sum()), len(evu)
                l_, h_ = wilson(k, n)
                uq.append({"period": per.replace("_", "–"), "event": ev, "rule": label, "events": n,
                           "without_permit": k, "share": k / n if n else np.nan, "ci_low": l_, "ci_high": h_})
            if ev == "new":   # by ZIP, unit rule
                pass
    uq = pd.DataFrame(uq)
    uq.to_csv(OUT / "rq3_unpermitted.csv", index=False)
    # by ZIP (both periods pooled, unit rule)
    zrows = []
    for per, (s, e) in PERIODS.items():
        lo = (pd.Timestamp(s) - pd.DateOffset(years=3))
        r_ = rel[(rel.permitissuedate >= lo) & (rel.permitissuedate < e)]
        has = set(zip(r_.zip, r_.unit))
        for ev in ("new", "popup"):
            evu = base[base[f"cov{per}"] & base[f"L{per}_{ev}"].fillna(False).astype(bool)][["zip", "unit"]]
            evu = evu.assign(permit=[(z, u) in has for z, u in zip(evu.zip, evu.unit)], event=ev, period=per)
            zrows.append(evu)
    zr = pd.concat(zrows).groupby(["zip", "event"]).agg(events=("permit", "size"),
                                                        without_permit=("permit", lambda x: int((~x).sum())))
    zr["share"] = zr.without_permit / zr.events
    zr.reset_index().to_csv(OUT / "rq3_unpermitted_by_zip.csv", index=False)
    # ---- D. rough floor area and housing (covered units; assumptions in stage7_plan.md)
    fa = []
    lab = {}
    for per in PERIODS:
        lab[per] = pd.concat([pd.read_parquet(ROOT / f"data/interim/lidar_change_city/{z}_{per}.parquet")[
            ["unit", "bldg_m2_22", "h50_22", "frac_up", "dh_p50"]].assign(zip=z) for z in zips])
    new_fa = []
    for per, yrs in (("2018_2022", 4), ("2022_2025", 3)):
        d = base[base[f"cov{per}"]].merge(lab[per], on=["zip", "unit"])
        n_ = d[d[f"L{per}_new"].fillna(False).astype(bool)]
        storeys = np.maximum(1, np.round(n_.h50_22 / STOREY_M))
        nfa = float((n_.bldg_m2_22 * storeys).sum())
        u_ = d[d[f"L{per}_popup"].fillna(False).astype(bool)]
        added = np.maximum(1, np.round(u_.dh_p50.clip(lower=0) / STOREY_M))
        pfa = float((u_.bldg_m2_22 * u_.frac_up * added).sum())
        cov = float(base[f"cov{per}"].mean())
        fa.append({"period": per.replace("_", "–"), "covered_share": cov, "new_units_detected": len(n_),
                   "new_floor_area_m2": nfa, "popup_units_detected": len(u_), "popup_added_floor_area_m2": pfa,
                   "new_floor_area_m2_scaled_to_all_units": nfa / cov, "popup_floor_area_m2_scaled": pfa / cov})
        new_fa.append(n_.assign(fa=n_.bldg_m2_22 * storeys)[["zip", "unit", "fa"]])
    # average floor area per housing unit from new-construction permits that state a unit count
    nf = pd.concat(new_fa)
    cal = pu[pu.is_new & pd.to_numeric(pu.numberofunits, errors="coerce").gt(0)].copy()
    cal["nu"] = pd.to_numeric(cal.numberofunits, errors="coerce")
    cal = cal.sort_values("permitissuedate").drop_duplicates(["zip", "unit"]).merge(nf, on=["zip", "unit"])
    m2_per_home = float(cal.fa.sum() / cal.nu.sum()) if len(cal) else np.nan
    for r in fa:
        r["m2_per_home_from_permits"] = m2_per_home
        r["homes_new_construction_scaled"] = r["new_floor_area_m2_scaled_to_all_units"] / m2_per_home
        r["homes_popups_upper_bound_scaled"] = r["popup_floor_area_m2_scaled"] / m2_per_home
    fa = pd.DataFrame(fa)
    fa.to_csv(OUT / "rq3_floor_area_housing.csv", index=False)
    S["calibration_permits_with_unit_count"] = int(len(cal))
    S["m2_per_home"] = m2_per_home
    # ---- figure: cumulative share detected by years after issue
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (_, r), c in zip(tt.iterrows(), (PALETTE["primary"], PALETTE["accent"], PALETTE["positive"], PALETTE["neutral"])):
        ks = [k for k in range(0, 7) if r.issue_year + k <= 2025]
        ax.plot(ks, [r[f"detected_within_{k}y"] for k in ks], "-o", color=c, label=f"issued {int(r.issue_year)} (n={int(r.units)})")
    ax.set(xlabel="years after permit issue", ylabel="share of units with new construction detected",
           title="New-construction permits: time to visible construction\n"
                 "(observed 2019, 2020, 2022–2025; 2021 not observed, so steps are flat there)", ylim=(0, 1))
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "rq3_time_to_detection.png", dpi=170)
    plt.close(fig)
    S.update({"permit_to_construction": a.to_dict("records"), "unpermitted": uq.to_dict("records"),
              "floor_area": fa.to_dict("records")})
    (OUT / "rq3_summary.json").write_text(json.dumps(S, indent=1, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(json.dumps({k: S[k] for k in ("permits", "permits_linked_to_main_units", "link_by_opa_share",
                                         "calibration_permits_with_unit_count", "m2_per_home")}, indent=1))
    print(a.round(3).to_string(index=False))
    print(tt.round(3).to_string(index=False))
    print(uq.round(3).to_string(index=False))
    print(fa.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
