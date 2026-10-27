"""Citywide study-area mask from the CURRENT zoning base districts (data/raw/zoning/zoning_basedistricts.gpkg).

    python scripts/study_area.py   -> data/interim/study_area_mask.gpkg (layers: districts, mask), study_area_stats.csv

Kept: residential and low/mid-density mixed-use districts (KEEP below). Removed: industrial (I-1, I-2, I-3, I-P,
ICMX), auto-oriented commercial (CA-1, CA-2), high-density centre (CMX-4, CMX-5), every special-purpose district
(airport, parks, institutions, civic, entertainment, stadiums) and the Philadelphia Navy Yard (NAVY_YARD), whose
redeveloped core is zoned CMX-3. Water is not zoned and so falls outside the mask. Current zoning (not 2018) is
used so that lots rezoned from industrial to residential stay in. Decisions: results/citywide/study_area.md.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/interim"
KEEP = {"RSD-1", "RSD-2", "RSD-3", "RSA-1", "RSA-2", "RSA-3", "RSA-4", "RSA-5", "RSA-6", "RTA-1", "RTA-2",
        "RM-1", "RM-2", "RM-3", "RM-4", "RMX-1", "RMX-2", "RMX-3", "IRMX",
        "CMX-1", "CMX-2", "CMX-2.5", "CMX-3"}
# Navy Yard: zoning polygons whose centroid lies south of I-95 between the Schuylkill and the Delaware
NAVY_YARD = box(-75.200, 39.880, -75.150, 39.900)
# area types used to stratify ZIP codes
TYPE = {"detached": {"RSD-1", "RSD-2", "RSD-3"},
        "rowhouse": {"RSA-1", "RSA-2", "RSA-3", "RSA-4", "RSA-5", "RSA-6", "RTA-1", "RTA-2", "RM-1"},
        "mixed": {"RM-2", "RM-3", "RM-4", "RMX-1", "RMX-2", "RMX-3", "IRMX", "CMX-1", "CMX-2", "CMX-2.5", "CMX-3"}}


def build():
    z = gpd.read_file(ROOT / "data/raw/zoning/zoning_basedistricts.gpkg").to_crs(2272)
    ny = gpd.GeoSeries([NAVY_YARD], crs=4326).to_crs(2272).iloc[0]
    z["navy_yard"] = z.geometry.centroid.within(ny)
    z["kept"] = z.long_code.isin(KEEP) & ~z.navy_yard
    z["area_type"] = z.long_code.map({c: t for t, cs in TYPE.items() for c in cs})
    z["km2"] = z.area * 0.09290304 / 1e6
    kept = z[z.kept].copy()
    kept["geometry"] = kept.geometry.buffer(0)
    mask = gpd.GeoDataFrame(geometry=[kept.union_all()], crs=2272)
    kept[["long_code", "zoninggroup", "area_type", "km2", "geometry"]].to_file(OUT / "study_area_mask.gpkg",
                                                                               layer="districts", driver="GPKG")
    mask.to_file(OUT / "study_area_mask.gpkg", layer="mask", driver="GPKG")
    stats = z.groupby(["long_code", "kept"]).agg(km2=("km2", "sum"), polygons=("km2", "size"),
                                                 navy_yard_km2=("km2", lambda s: s[z.loc[s.index, "navy_yard"]].sum()))
    stats.round(3).to_csv(ROOT / "results/citywide/study_area_stats.csv")
    print(f"kept {kept.km2.sum():.1f} km2 of {z.km2.sum():.1f} km2 zoned ({len(kept)} of {len(z)} polygons); "
          f"Navy Yard removed {z[z.navy_yard & z.long_code.isin(KEEP)].km2.sum():.2f} km2 of kept codes")
    print(kept.groupby("area_type").km2.sum().round(1).to_string())
    return kept, mask


if __name__ == "__main__":
    (ROOT / "results/citywide").mkdir(parents=True, exist_ok=True)
    build()
    sys.exit(0)
