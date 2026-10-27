"""Stage 7 downloads (results/citywide/stage7_plan.md). Public data only, no keys; sources written next to the files.

    python scripts/stage7_data.py [permits] [acs] [tiger] [gtfs]     # default: all

permits  Carto `permits` (OpenDataPhilly), study-area bounding box, issued 2014-01-01..2025-12-31, building / zoning /
         demolition / administrative types (ECLIPSE names and Hansen codes); one query per year; per-year coverage of
         OPA account and coordinates is printed (Carto had a transient location outage on 2026-10-21)
         -> data/raw/li/permits_studyarea_2014_2025.csv
acs      ACS 5-year summary files (no API key): 2019-2023 table-based files, 2015-2019 Pennsylvania sequence files;
         B01003 population, B19013 median household income (+ MOE), B25001 housing units, B25064 median gross rent;
         Philadelphia block groups -> data/raw/acs/acs5_{2019,2023}_bg_philadelphia.csv
tiger    2020 block groups, Pennsylvania -> data/raw/boundaries/tl_2020_42_bg.zip
gtfs     SEPTA GTFS (www3.septa.org/developer) -> data/raw/transit/gtfs_public.zip
"""

import io
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data/raw"
ACS = "https://www2.census.gov/programs-surveys/acs/summary_file"
TABLES = ("B01003", "B19013", "B25001", "B25064")
STAMP = f"downloaded {pd.Timestamp.now():%Y-%m-%d}"


def get(url, **kw):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=600, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {attempt + 1}: {e}", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(url)


def permits():
    import tiles as T
    from permits import CARTO, FIELDS
    zt = pd.read_csv(ROOT / "results/citywide/splits_citywide_zips.csv", dtype={"zip": str})
    x0, y0, x1, y1 = T.study_area(tuple(zt.zip), buffer_ft=0).to_crs(4326).total_bounds
    types = ("'Residential Building', 'Building', 'Zoning', 'Demolition', 'Administrative', "
             "'Site / Utility Permit'")
    parts = []
    for y in range(2014, 2026):
        sql = (f"SELECT {FIELDS} FROM permits WHERE permitissuedate >= '{y}-01-01' AND permitissuedate < '{y + 1}-01-01' "
               f"AND (permittype IN ({types}) OR permittype LIKE 'BP\\_%' OR permittype LIKE 'ZP\\_%') "
               f"AND the_geom && ST_MakeEnvelope({x0}, {y0}, {x1}, {y1}, 4326)")
        p = pd.read_csv(io.StringIO(get(CARTO, params={"q": sql, "format": "csv"}).text),
                        dtype={"opa_account_num": str, "parcel_id_num": str, "zip": str})
        print(f"  {y}: {len(p):,} permits, OPA {p.opa_account_num.notna().mean():.1%}, "
              f"coords {p.lat.notna().mean():.1%}", flush=True)
        parts.append(p)
    out = RAW / "li/permits_studyarea_2014_2025.csv"
    pd.concat(parts).to_csv(out, index=False)
    (RAW / "li/SOURCE_permits_studyarea.txt").write_text(
        f"{CARTO} table=permits; issued 2014-01-01..2025-12-31; bbox {[x0, y0, x1, y1]} (EPSG:4326, study area); "
        f"types: building / zoning / demolition / administrative / site, Hansen BP_* and ZP_*; {STAMP}\n")


def acs():
    d = RAW / "acs"
    d.mkdir(parents=True, exist_ok=True)
    # 2019-2023: table-based summary files (all US geographies; keep Philadelphia block groups)
    t23 = None
    for t in TABLES:
        x = pd.read_csv(io.BytesIO(get(f"{ACS}/2023/table-based-SF/data/5YRData/acsdt5y2023-{t.lower()}.dat").content),
                        sep="|", dtype=str)
        x = x[x.GEO_ID.str.startswith("1500000US42101")]
        keep = {"GEO_ID": "geoid", f"{t}_E001": t, f"{t}_M001": f"{t}_moe"}
        x = x[list(keep)].rename(columns=keep)
        t23 = x if t23 is None else t23.merge(x, on="geoid")
    t23["geoid"] = t23.geoid.str[-12:]
    t23.to_csv(d / "acs5_2023_bg_philadelphia.csv", index=False)
    # 2015-2019: Pennsylvania sequence files (tracts and block groups); estimates e*, margins m*
    base = f"{ACS}/2019/data/5_year_seq_by_state/Pennsylvania/Tracts_Block_Groups_Only"
    lk = pd.read_csv(io.StringIO(get(f"{ACS}/2019/documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt")
                                 .text), dtype=str)
    geo = pd.read_csv(io.BytesIO(get(f"{base}/g20195pa.csv").content), header=None, dtype=str, encoding="latin-1")
    geo = geo[[4, 48]].rename(columns={4: "logrecno", 48: "geoid_full"})   # LOGRECNO, GEOID
    geo = geo[geo.geoid_full.str.startswith("15000US42101")]
    t19 = geo.assign(geoid=geo.geoid_full.str[-12:])[["logrecno", "geoid"]]
    for t in TABLES:
        row = lk[(lk["Table ID"] == t) & lk["Start Position"].notna()].iloc[0]
        seq, start = row["Sequence Number"], int(row["Start Position"])
        z = zipfile.ZipFile(io.BytesIO(get(f"{base}/20195pa{seq}000.zip").content))
        vals = {}
        for kind in ("e", "m"):
            f = pd.read_csv(z.open(f"{kind}20195pa{seq}000.txt"), header=None, dtype=str)
            # columns: FILEID, FILETYPE, STUSAB, CHARITER, SEQUENCE, LOGRECNO, then the cells; start is 1-based
            vals[kind] = f[[5, start - 1]].rename(columns={5: "logrecno", start - 1: t if kind == "e" else f"{t}_moe"})
        t19 = t19.merge(vals["e"], on="logrecno").merge(vals["m"], on="logrecno")
    t19.drop(columns="logrecno").to_csv(d / "acs5_2019_bg_philadelphia.csv", index=False)
    (d / "SOURCE.txt").write_text(
        f"ACS 5-year 2019-2023: {ACS}/2023/table-based-SF/data/5YRData/acsdt5y2023-<table>.dat\n"
        f"ACS 5-year 2015-2019: {base}/20195pa<seq>000.zip + g20195pa.csv; sequence lookup "
        f"{ACS}/2019/documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt\n"
        f"Tables {', '.join(TABLES)}; Philadelphia County block groups; {STAMP}\n")
    print(f"  ACS 2019: {len(t19)} block groups; ACS 2023: {len(t23)} block groups", flush=True)


def tiger():
    u = "https://www2.census.gov/geo/tiger/TIGER2020/BG/tl_2020_42_bg.zip"
    (RAW / "boundaries/tl_2020_42_bg.zip").write_bytes(get(u).content)
    with open(RAW / "boundaries/SOURCES.txt", "a", encoding="utf-8") as fh:
        fh.write(f"tl_2020_42_bg.zip: {u}; {STAMP}\n")


def gtfs():
    u = "https://www3.septa.org/developer/gtfs_public.zip"
    (RAW / "transit").mkdir(parents=True, exist_ok=True)
    (RAW / "transit/gtfs_public.zip").write_bytes(get(u).content)
    (RAW / "transit/SOURCE.txt").write_text(f"SEPTA GTFS: {u}; {STAMP}\n")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["permits", "acs", "tiger", "gtfs"]
    for s in steps:
        print(f"== {s}", flush=True)
        globals()[s]()
    print("DATA DONE", flush=True)
