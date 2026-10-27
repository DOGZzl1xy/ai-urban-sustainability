# Exclusion check of the cloud package

Run 2026-10-26 21:22 by `cloud/pack.py` on the packed files (git HEAD cca94d2).

**Result: PASS**

| Check | Result |
|---|---|
| Packed patches (unique) | 2,340 in 37 ZIPs; 12,736 pairs over 6 periods |
| Patches intersecting a sealed test ZIP | 0 |
| Patches intersecting a test ZIP buffered by 100 m | 0 |
| Packed file paths naming a test ZIP (19106, 19116, 19127, 19134, 19136, 19139, 19150, 19153) | 0 |
| Packed paths from rasters, LAS, imagery, unit polygons, review pages or labels | 0 |
| Address columns in the exported tables | 0 |

Geometry: each patch box from `m4_pair_index.csv` (EPSG:2272 ft) against the 2019 ZIP polygons of the eight test ZIPs, unbuffered and buffered by 100 m.
