"""L&I permits around the study area (Carto `permits`), selected by location so cross-ZIP neighbours are included."""

from __future__ import annotations

import io
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "li"
CARTO = "https://phl.carto.com/api/v2/sql"
FIELDS = ("permitnumber, parcel_id_num, opa_account_num, address, zip, permittype, permitdescription, typeofwork, "
          "approvedscopeofwork, commercialorresidential, permitissuedate, permitcompleteddate, status, "
          "certificateofoccupancydate, numberofstories, numberofunits, zoningpermitjobid, systemofrecord, "
          "ST_Y(the_geom) AS lat, ST_X(the_geom) AS lng")

# Permit vocabularies differ by system of record: ECLIPSE (2020-03 onward, and some earlier records) uses names,
# Hansen (to 2020-03) uses codes such as BP_NEWCNST / BP_ADDITON / BP_ALTER / ZP_ZON/USE. Both are mapped here.
BUILDING_TYPES = {"Residential Building", "Building"}
NEW = {"New Construction"}
ADDITION = {"Addition and/or Alteration", "Addition and/or Alterations"}
ZONING_DEV = {"New construction, addition, GFA change", "Combined Lot Line Relocation and New Development"}
HANSEN_BUILDING = {"BP_NEWCNST", "BP_ADDITON", "BP_ALTER"}
HANSEN_ZONING_DEV = {("ZP_ZON/USE", "NEWCON"), ("ZP_ZON/USE", "ADD"), ("ZP_ZONING", "SFADD")}
# any of these on the parcel or a neighbour counts as "a related permit exists" for the unpermitted screen;
# trade permits (electrical, plumbing, mechanical, fire suppression) and EZ minor permits do not
RELATED_TYPES = BUILDING_TYPES | HANSEN_BUILDING | {"Zoning", "Demolition", "Site / Utility Permit", "Administrative",
                                                   "BP_DEMO", "BP_ADMINST"}

# upward needs an explicit vertical marker; a "two story addition" alone is usually a rear addition
UPWARD = (r"(?i)((2nd|second|3rd|third|4th|fourth)[- ](story|storey|floor) (rooftop )?addition|addition (above|atop|on top of|"
          r"on the roof|on roof)|vertical addition|pop.?up|(add|adding|addition of) (a |an )?(new )?(2nd|second|3rd|third|"
          r"4th|fourth|additional)[- ]?(story|storey|floor)|penthouse)")
REAR = (r"(?i)(rear|back|side)[- ](yard )?(addition|extension)|addition (at|to|in) the (rear|back)|story rear|"
        r"rear (\w+[- ]){1,3}addition")


def download(area: gpd.GeoDataFrame, start="2014-01-01", end="2023-01-01", name="study") -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    f = RAW / f"permits_{name}_{start[:4]}_{end[:4]}.csv"
    if not f.exists():
        x0, y0, x1, y1 = area.to_crs(4326).total_bounds
        sql = (f"SELECT {FIELDS} FROM permits WHERE permitissuedate >= '{start}' AND permitissuedate < '{end}' "
               f"AND the_geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 4326)")
        for attempt in range(4):
            try:
                r = requests.get(CARTO, params={"q": sql, "format": "csv"}, timeout=600)
                r.raise_for_status()
                break
            except requests.RequestException:
                time.sleep(10 * (attempt + 1))
        f.write_text(r.text, encoding="utf-8")
        (RAW / "SOURCE_permits.txt").write_text(f"{CARTO} table=permits bbox={[x0, y0, x1, y1]} {start}..{end}; "
                                                f"downloaded {pd.Timestamp.now():%Y-%m-%d}\n")
    p = pd.read_csv(f, dtype={"opa_account_num": str, "parcel_id_num": str, "zip": str})
    for c in ["permitissuedate", "permitcompleteddate", "certificateofoccupancydate"]:
        p[c] = pd.to_datetime(p[c], errors="coerce", utc=True).dt.tz_convert(None)
    p["opa"] = p.opa_account_num.str.replace(r"\.0$", "", regex=True).str.zfill(9).where(p.opa_account_num.notna())
    return p


def classify(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    bldg = p.permittype.isin(BUILDING_TYPES)
    pt, tw = p.permittype.fillna(""), p.typeofwork.fillna("")
    text = (p.approvedscopeofwork.fillna("") + " " + p.permitdescription.fillna(""))
    p["is_new"] = (bldg & tw.isin(NEW)) | ((pt == "BP_NEWCNST") & tw.isin({"ENTIRE", "FOUND"}))
    p["is_addition"] = (bldg & tw.isin(ADDITION)) | (pt == "BP_ADDITON") | ((pt == "BP_ALTER") & (tw == "MAJOR"))
    p["is_upward"] = p.is_addition & text.str.contains(UPWARD)
    p["is_rear"] = p.is_addition & text.str.contains(REAR)  # a permit can be both
    p["is_zoning_dev"] = (((pt == "Zoning") & tw.isin(ZONING_DEV))
                          | pd.Series(list(zip(pt, tw)), index=p.index).isin(HANSEN_ZONING_DEV))
    p["is_related"] = pt.isin(RELATED_TYPES) | pt.str.startswith("ZP_")
    return p
