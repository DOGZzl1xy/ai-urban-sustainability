# QC — LiDAR epochs and permits (ZIP 19146)

## LiDAR

| | 2018 | 2022 | 2025 |
|---|---|---|---|
| Source | PASDA `phillyLiDAR/LAS2018` (zipped LAS) | PASDA `phillyLiDAR/2022/LAS/Classified_LAS` | PASDA `phillyLiDAR/2025/LAS` (posted 2026-01-12/15) |
| Tiles (study area + 300 ft) | 20 | 14 | 17 |
| Format / CRS | LAS 1.4, point format 6, CRS NAD83 / Pennsylvania South (ftUS) + NAVD88 height - Geoid12B (ftUS) | LAS 1.4, point format 6, CRS NAD83(2011) / Pennsylvania South (ftUS) + NAVD88 height (ftUS) - US Geoid Model of 2018 | LAS 1.4, point format 6, CRS NAD83(2011) / Pennsylvania South (ftUS); x/y PA South US ft, z US ft (ground matches 2018) |
| **Acquisition** (GPS time) | **2018-04-14 → 2018-04-22** | **2022-03-29 → 2022-03-30** | **2025-04-18 → 2025-04-18** |
| Flight days in study tiles | 2018-04-14, 2018-04-22 | 2022-03-29, 2022-03-30 | 2025-04-18 |
| Median density (tile, classes 7/12/18 dropped) | 43.1 pts/m² | 38.6 pts/m² | 29.6 pts/m² |
| Class shares | {'1': 0.35, '2': 0.1501, '3': 0.0209, '4': 0.0172, '5': 0.0477, '6': 0.3782, '7': 0.0043, '9': 0.0012, '10': 0.0, '12': 0.0202, '17': 0.0103} | {'1': 0.7695, '2': 0.2083, '7': 0.0003, '8': 0.0151, '9': 0.0004, '17': 0.0064, '18': 0.0, '20': 0.0} | {'1': 0.1704, '2': 0.2587, '3': 0.0323, '4': 0.0305, '5': 0.1473, '6': 0.3382, '7': 0.0107, '9': 0.0039, '17': 0.0079, '18': 0.0, '20': 0.0001} |

Classes used: 2 ground (DTM); everything else except noise (7, 18) and overlap (12) for the DSM. Vegetation comes
from return structure in every epoch (multi-return pulses; masked when they are the majority in a 0.5 m cell),
because the 2022 "Classified_LAS" only separates ground. 2018 and 2025 also carry building (6) and vegetation classes.

CRS: 2018 is EPSG:2272 and 2022 is EPSG:6565 (both PA South, US ft; different NAD83 realisations, cm-level). The
2025 headers carry no CRS.

**2025 acquisition date.** PASDA's data summary (dataset 7158) and the BTAA record give the acquisition as
"April 2022" and repeat the 2022 description. The point GPS times (adjusted standard GPS time) show that every
tile in the study area was flown on the date in the table above, and the LAS headers were written in 2025 (day 196).
The 2022 metadata was evidently copied over; the point timestamps are the date used in this project.

**Truncated 2025 files on PASDA** (shorter than their headers; the server's Content-Length is the truncated size):
26822E227832N.las: 7,088,269 of 11,993,334 points. Whole points are read; the gap shows as missing coverage.

## Vertical registration (ground, 1.5 m cells with ground returns in both epochs)

- 2022 − 2018: 1,784,859 cells; median difference **+0.030 m**, IQR 0.047 m. Per ~300 m block (100 blocks with > 500 cells): median of block medians +0.029 m, range -0.015 to +0.056 m.
- 2025 − 2018: 1,788,389 cells; median difference **-0.012 m**, IQR 0.052 m. Per ~300 m block (92 blocks with > 500 cells): median of block medians -0.009 m, range -0.037 to +0.024 m.

nDSM is computed against each epoch's own DTM, so a vertical offset of this size cancels out of ΔnDSM; it is about
1–2% of the 2.5 m change threshold.

## Permit location coverage (permits in the study bounding box)

|   permitissuedate |    n |   opa |   xy |
|------------------:|-----:|------:|-----:|
|              2014 | 4623 | 0.929 |    1 |
|              2015 | 5410 | 0.959 |    1 |
|              2016 | 6181 | 0.971 |    1 |
|              2017 | 6760 | 0.975 |    1 |
|              2018 | 6703 | 0.983 |    1 |
|              2019 | 6795 | 0.985 |    1 |
|              2020 | 4654 | 0.982 |    1 |
|              2021 | 4843 | 0.978 |    1 |
|              2022 | 4013 | 0.974 |    1 |

Older (pre-2020, Hansen) and newer (ECLIPSE) permits both carry OPA numbers and coordinates in the current Carto table.
