# Citywide study area (frozen 2026-10-23)

Mask: `data/interim/study_area_mask.gpkg`. Script: `scripts/study_area.py`. Per-code areas: `study_area_stats.csv`.

**Source.** City of Philadelphia, Zoning Base Districts. This is the current version, from the ArcGIS FeatureServer `Zoning_BaseDistricts`, layer last edited 2026-10-21 and downloaded 2026-10-23; 29,208 polygons. Current zoning is used instead of 2018 zoning, so lots rezoned from industrial to residential stay in the study area.

## Kept: residential and low/mid-density mixed use (152.5 km² of 280.1 km² zoned)

| Area type | Codes | km² |
|---|---|---|
| Rowhouse / attached | RSA-1 … RSA-6, RTA-1, RTA-2, RM-1 | 100.0 |
| Detached | RSD-1, RSD-2, RSD-3 | 28.2 |
| Mixed | RM-2, RM-3, RM-4, RMX-1, RMX-2, RMX-3, IRMX, CMX-1, CMX-2, CMX-2.5, CMX-3 | 24.3 |

IRMX (industrial-residential mixed use, 0.77 km²) is kept by decision of 2026-10-23: it holds many industrial-to-residential conversions.

## Removed

- **Industrial:** I-1, I-2, I-3, I-P; also ICMX (industrial-commercial mixed use; residential mostly not permitted).
- **High-density centre:** CMX-4, CMX-5.
- **Auto-oriented commercial:** CA-1, CA-2.
- **Every special-purpose district:**
  - SP-AIR (airport)
  - SP-PO-A, SP-PO-P (parks and open space)
  - SP-INS (institutional campuses)
  - SP-CIV, SP-ENT, SP-STA
- **Navy Yard.** Its redeveloped core is zoned CMX-3, so it is removed by location: every zoning polygon whose centroid lies in lon −75.200 to −75.150, lat 39.880 to 39.900 (south of I-95, between the Schuylkill and the Delaware). That removes 2.23 km² of otherwise kept codes.
- **Water:** not zoned, so it falls outside the mask.

## High-rise flag (applied once citywide LiDAR is processed)

Grid cells (100 m, 250 m) and block groups where the 90th percentile of building height (nDSM on building cells) exceeds **25 m** in 2018 or 2025 are flagged and **left out of the main results**. They are reported separately.

## Implementation notes (added 2026-10-26; the rule above is unchanged)

- Script `scripts/highrise_flag.py`. Building cell = nDSM > 2.5 m and not majority vegetation (the change rules'
  definition), counted inside the mask and inside each ZIP once; heights binned at 0.25 m and merged across ZIPs.
- 2025 uses the corrected vegetation (`veg2025_fix.md`). A first run with the old 2025 vegetation flagged many wooded
  cells through tree canopy; it is kept as `highrise_trainval_2025_returns_only.csv` and not used.
- Train / val ZIPs now (`highrise_trainval.csv`); the test ZIPs are added at the round-2 test, and cells and block
  groups that straddle a test ZIP are recomputed then from both parts.
