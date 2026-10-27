"""Permit -> unit matching, recall sets, and unpermitted-change candidates."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd


def match_permits(p: pd.DataFrame, units: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach a unit id to each permit: OPA account first, then the permit point falling inside a unit."""
    opa2unit = units[["unit", "opas"]].explode("opas").dropna().drop_duplicates("opas").set_index("opas").unit
    p = p.copy()
    p["unit"] = p.opa.map(opa2unit)
    p["match"] = np.where(p.unit.notna(), "opa", None)
    miss = p.unit.isna() & p.lat.notna()
    if miss.any():
        pts = gpd.GeoDataFrame(p.loc[miss, []], geometry=gpd.points_from_xy(p.loc[miss, "lng"], p.loc[miss, "lat"]),
                               crs=4326).to_crs(2272)
        j = gpd.sjoin(pts, units[["unit", "geometry"]], how="left", predicate="within")
        j = j[~j.index.duplicated()]
        p.loc[miss, "unit"] = j.unit
        p.loc[miss & p.unit.notna(), "match"] = "point"
    return p


def neighbours(units: gpd.GeoDataFrame, dist_ft=3.0) -> dict:
    """unit -> set of units touching it (within dist_ft)."""
    b = units[["unit", "geometry"]].copy()
    b["geometry"] = b.buffer(dist_ft)
    j = gpd.sjoin(b, units[["unit", "geometry"]], predicate="intersects")
    j = j[j.unit_left != j.unit_right]
    return j.groupby("unit_left").unit_right.apply(set).to_dict()


def recall_sets(p: pd.DataFrame, d18, d22) -> pd.DataFrame:
    """One row per (unit, permit type) whose work should show up between the two LiDAR epochs.

    primary: issued after the 2018 flight and completed before the 2022 flight, so the work must fall between epochs.
    completed_in_window: completed in (d18, d22) regardless of issue date. This was the set first evaluated; it mixes
      in permits issued before 2018 whose work was already standing in the 2018 data (Hansen completion dates are
      often administrative close-outs). Reported alongside for transparency.
    secondary: no completed date, issued in (d18 - 3y, d22 - 12 months].
    """
    rows = []
    for kind, mask in [("new", p.is_new), ("upward", p.is_upward), ("rear", p.is_rear)]:
        q = p[mask & p.unit.notna()]
        done = (q.permitcompleteddate > d18) & (q.permitcompleteddate < d22)
        prim = q[done & (q.permitissuedate > d18)]
        comp = q[done]
        sec = q[q.permitcompleteddate.isna() & (q.permitissuedate > d18 - pd.DateOffset(years=3))
                & (q.permitissuedate <= d22 - pd.DateOffset(months=12))]
        for name, s in (("primary", prim), ("completed_in_window", comp), ("secondary", sec)):
            s = s.sort_values("permitissuedate").drop_duplicates("unit")
            rows.append(s.assign(kind=kind, set=name))
    return pd.concat(rows, ignore_index=True)


def unpermitted(flags: pd.DataFrame, p: pd.DataFrame, nb: dict, d18, d22) -> pd.DataFrame:
    """Units flagged as new / pop-up / rear with no related permit on the unit or a neighbour in [d18 - 3y, d22]."""
    win = p[p.is_related & p.unit.notna() & (p.permitissuedate >= d18 - pd.DateOffset(years=3))
            & (p.permitissuedate <= d22)]
    has = set(win.unit.astype(int))
    f = flags[flags.chg_any_build].copy()
    f["permit_on_unit"] = f.unit.isin(has)
    f["permit_on_neighbour"] = [bool(nb.get(u, set()) & has) for u in f.unit]
    f["candidate"] = ~f.permit_on_unit & ~f.permit_on_neighbour
    return f


def bootstrap_rate(x: np.ndarray, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) == 0:
        return np.nan, np.nan
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
