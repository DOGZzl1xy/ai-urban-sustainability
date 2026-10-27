# 2025 vegetation fix (found 2026-10-26, before any 2025 citywide label was used)

## Problem

Vegetation is separated from roofs by return structure: a cell is vegetation when most of its returns come from
multi-return pulses (leaf-off canopy splits pulses, roofs do not). The 2025 flight (18 April 2025, leaves coming out)
breaks this: in three classified 2025 tiles, **42–58% of the points classed as high vegetation are single returns**
(buildings: 89–91%). Canopy then counts as "building" in 2025.

Found through the high-rise flag: of the 100 m cells flagged (p90 building height > 25 m), 42% exceeded 25 m only in
2025. In wooded ZIP 19118, building cells above 25 m were 149 k (2018), 116 k (2022) and 570 k (2025).

## Test of the fix (count ASPRS classes 3/4/5 as vegetation as well, 2025 only; it is the only fully classified year)

Share of cells with majority vegetation, same three wooded tiles in 19118:

| | Tile A | Tile B | Tile C |
|---|---|---|---|
| 2018 | 0.42 | 0.36 | 0.53 |
| 2022 | 0.39 | 0.33 | 0.51 |
| 2025, return structure only | 0.34 | 0.29 | 0.45 |
| 2025, + classes 3/4/5 | 0.42 | 0.33 | 0.53 |

2022→2025 labels in the rowhouse ZIPs 19146 / 19147 (units with full coverage) barely move: new construction
159 → 153 and 20 → 19, pop-ups 165 → 161 and 72 → 69; the fix only removes detections, it adds none (one pop-up added
in 19147). The effect is concentrated in wooded areas and in the high-rise flag.

## What changes

- `rasters.tile_rasters(..., veg_classes)`; `stream_lidar.py` uses classes 3/4/5 for 2025 only. 2025 is streamed
  again (all 548 tiles); 2025 rasters, the 2022→2025 labels and the high-rise flag are rebuilt.
- The earlier 2025 outputs are kept for the record: `data/interim/citywide_tiles/2025_returns_only/`,
  `data/interim/lidar_change_city_2025_returns_only/`, `results/citywide/processing/stream_2025_returns_only.csv`,
  `highrise_trainval_2025_returns_only.csv`.
- **M4 is not affected:** it trains on periods up to 2022 only.
- **Round-1 final test** (2022→2025, run once) used the return-structure 2025 labels. It is not rerun; the report notes
  that its truth for 19146 / 19147 holds about 3% more detections than the fixed rule gives.
- **Spot-check:** the first draw (made just before this was found; no one had seen the page) included 2022→2025
  detections under the old rule. It was deleted and is redrawn, unchanged in design and seed, after the rebuild.

## Result of the rebuild (2026-10-26 06:52; 548 tiles re-streamed, 0 failures)

Train / val ZIPs, units in the study area with full coverage:

| 2022→2025 | Return structure only | + classes 3/4/5 | Change |
|---|---|---|---|
| New construction | 1,502 | 1,346 | −10% |
| Pop-up | 807 | 762 | −6% |
| Demolition | 626 | 713 | +14% |

| High-rise flag (p90 > 25 m) | Before | After |
|---|---|---|
| 100 m cells flagged | 1,166 | 731 |
| of which only in 2025 | 486 | 51 |
| 250 m cells / block groups | 244 / 37 | 159 / 35 |

Demolitions rise: where 2022 canopy over a roof still counted as building (2022 has no vegetation classes), the 2025
canopy is now vegetation, so cover can drop. Demolition is not an M4 label; it is reported with this caveat if used
in RQ2. Periods ending before 2025 are unchanged.
