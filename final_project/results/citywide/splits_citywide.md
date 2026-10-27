# Second-round split: sealed test ZIPs, train and validation (frozen 2026-10-23, before any citywide data)

Script: `scripts/citywide_split.py` (seed 20261023). Outputs:
- `data/interim/splits_citywide.gpkg` (layers `zips`, `blocks`)
- `splits_citywide.csv` (1 km blocks)
- `splits_citywide_zips.csv`

This split is **not changed again**.

## Sealed test ZIPs (second-round final test only)

Each ZIP is typed by the area type covering most of its study area (`zip_area_types.csv`).
- **Eligible:** at least 0.5 km² of study area; not one of the 7 Stage 3 training ZIPs (19132, 19121, 19143, 19125, 19130, 19123, 19122); not the round-1 test ZIPs 19146 and 19147.
- **Draw:** per type, round(22% × eligible), at least 1, drawn from the sorted list.

| Type | Eligible | Drawn | Test ZIPs (study-area km²) |
|---|---|---|---|
| Rowhouse | 28 | 6 | 19127 (0.53), 19134 (3.06), 19136 (4.16), 19139 (2.79), 19150 (2.81), 19153 (2.56) |
| Detached | 4 | 1 | 19116 (6.49) |
| Mixed | 2 | 1 | 19106 (0.68) |
| **Total** | 34 | **8** | 23.1 km², 15% of the study area |

Note: only two eligible ZIPs are mainly mixed use, so the mixed type rests on one small test ZIP (19106). Results by area type will therefore also be reported for units grouped by their own zoning type, not only by ZIP type.

## Train / validation (all other study-area land, including the Stage 3 ZIPs and 19146 / 19147)

- **1 km blocks:** the same grid as Stage 3, floor of coordinate ÷ 1 km in EPSG:2272.
- **Assignment:** a block whose study-area centre lies in a test ZIP is `test` (68 blocks, 23.0 km²). Of the rest, 15% are drawn as `val` (46 blocks, 20.8 km²); the others are `train` (259 blocks, 108.6 km²).
- **Patch rules** (applied when patches are cut):
  - a patch belongs to the block of its centre;
  - **a patch touching a test ZIP buffered by 100 m never enters training or validation**;
  - a train patch within 50 m of a val block is dropped.
- **Test evaluation** uses only units inside the test ZIPs.
- **19146 and 19147** (round-1 test) can enter training. Round 1 is closed, and their 2025 data cannot enter training anyway (see `M4_plan.md`).
- **Before M4 is frozen:** nothing in the test ZIPs is evaluated or visualised.
