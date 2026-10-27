# Stage 7: urban analysis (RQ2, RQ3) — fixed 2026-10-28, before any stage-7 data is downloaded

Inputs from stage 6 (`results/citywide/stage6/`, `stage6_plan.md`): LiDAR anchor periods 2018→2022 and 2022→2025
(new construction, pop-ups, demolition; units covered by LiDAR in both years) and model-dated new construction for
2019, 2020 (2018 anchor) and 2023, 2024 (2022 anchor). Pop-ups only from LiDAR. High-rise cells / block groups are
reported separately. Everything is aggregated; no address or parcel id is published.

## Data (downloaded for this stage)

| Data | Source | Use |
|---|---|---|
| L&I permits 2014–2025, study-area bounding box; building, zoning, demolition, administrative types only | Carto `permits` (OpenDataPhilly) | RQ3 |
| ACS 5-year 2015–2019 and 2019–2023, block groups: population, median household income (+ MOE), housing units, median gross rent | Census API | RQ2 context |
| 2020 block-group boundaries | TIGER/Line 2020 | interpolate 2019–2023 (2020 geography) onto 2019 block groups by area |
| SEPTA GTFS (bus and rail) | SEPTA `gtfs_public` | transit access |

Permit coverage (OPA account and coordinates) is checked per year before use (Carto had a transient location
outage on 2026-10-21).

## RQ2 — where, how much, when; transit and socioeconomic change

- **Rates, not raw counts:** events per 1,000 covered units per year, because LiDAR covers 64% of units in 2018→2022
  and 97% in 2022→2025, and the periods have different lengths (4 and 3 years).
- **Where:** maps at 250 m and block group; concentration (share of events in the top 10% of cells); by unit area
  type (rowhouse, mixed, detached) and ZIP.
- **When:** new construction by year — 2019, 2020 (model), 2022 (LiDAR) on the 2018-anchor unit set; 2023, 2024
  (model), 2025 (LiDAR) on the 2022-anchor unit set; annualised increments with the round-2 test error stated. Pop-ups:
  two LiDAR periods only.
- **Transit access:** distance from each cell centroid to the nearest rail-type stop (subway, elevated, trolley,
  regional rail; GTFS route types 0, 1, 2), bands < 400 m, 400–800 m, 800–1,600 m, > 1,600 m; bus stops within 400 m
  as a second measure. Rates by band.
- **Socioeconomic change (block groups):** baseline median household income (2015–2019) quintiles and change to
  2019–2023 (2020-geography values interpolated by area onto 2019 block groups; medians interpolated as area-weighted
  means — an approximation, stated). Rates by quintile.
- **Model:** negative binomial regression of event counts per block group with log(covered units × years) as offset;
  predictors: rail-distance band, baseline income quintile, income change, population change, area type share.
  Reported as associations with 95% intervals; no causal claim; ACS margins of error noted.

## RQ3 — permits and visible construction

- Permits classified with `src/permits.py` (new construction, addition, upward addition, demolition, related).
  Matched to units by OPA account (local unit table), else by point-in-polygon.
- **Permit → construction:** new-construction permits issued 2018-01 to 2020-12 → share whose unit shows LiDAR new
  construction by 2022, and by 2025 (units covered by LiDAR only). Same for addition permits vs LiDAR pop-ups.
- **Time to completion:** for new-construction permits issued 2018–2023 on units the model scores, the first year the
  unit is detected (2019, 2020, 2022, 2023, 2024, 2025), as a distribution of years after issue.
- **Change without a matching permit:** LiDAR new construction / pop-ups with no related permit on the unit's parcel
  issued from 3 years before the period start to the period end; share with a 95% interval. The pilot found very few,
  so the expected share is small; results are aggregated by ZIP, never listed by address.

## Floor area and housing (rough, assumptions stated)

New floor area = footprint area × storeys (median roof height ÷ 3 m) for new construction; footprint × added height ÷
3 m for pop-ups. Housing units: floor area ÷ an average unit size calibrated on permits that state a unit count.
Reported as orders of magnitude with the assumptions next to them.
