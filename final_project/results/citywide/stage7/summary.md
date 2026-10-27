# Stage 7: RQ2 and RQ3 results (2026-10-28)

Plan: `../stage7_plan.md`. Inputs: stage 6 (`../stage6/`), public downloads (permits, ACS summary files, TIGER 2020,
SEPTA GTFS; sources in `data/raw/*/SOURCE*.txt`). Study area: 520,903 units outside high-rise 250 m cells. All results
are aggregated; no address or parcel id is published. Scripts: `scripts/stage7_rq2.py`, `scripts/stage7_rq3.py`.

## RQ2 — where, how much, when

**How much (LiDAR, events per 1,000 units per year).** Compared on the 322,921 units covered in both periods (the 2018
flight covers fewer, more central units, so rates on all covered units overstate the 2018–2022 level):

| | 2018–2022 | 2022–2025 | change |
|---|---|---|---|
| New construction | 1.62 | 1.20 | −26% |
| Vertical additions (pop-ups) | 1.06 | 0.71 | −33% |
| Demolition | 0.62 | 0.58 | −6% |

**When (new construction, model years are estimates).** 2018 anchor (267,994 units): 2019 1.77, 2020 2.36, 2021–22
average 1.33 per 1,000 per year; 2022 anchor (420,037 units): 2023 0.71, 2024 1.07, 2025 1.02. The model under-counts
new construction by about 12% (round-2 test), so the 2021–22 residual (LiDAR minus model) is slightly inflated.
Construction peaked around 2019–2020 and ran at roughly half that pace in 2023–2025.

**Where.** Highly concentrated: the top 10% of 250 m cells hold 94% of new construction and 90% of pop-ups; 15% of
cells have any. Mixed-use units densify fastest (new 2.4 / 2.1, pop-ups 2.1 / 1.6 per 1,000 per year in the two
periods), rowhouse units next, detached areas almost not at all. Leading ZIPs: new construction 19122, 19121, 19146,
19123, 19125 (2018–22) and 19122, 19121, 19104, 19133, 19146 (2022–25); pop-ups 19146, 19130, 19125, 19121, 19122.
Maps: `figs/rq2_map_250m.png`.

**Transit.** Within 1.6 km of a rail-type stop (subway / elevated, trolley, regional rail) rates differ little;
beyond 1.6 km they collapse (new 0.14 vs 1.93, 2018–22). Negative binomial models per block group (945 block groups,
offset covered unit-years, controls for income, income change, population change, rowhouse share): beyond 1.6 km
IRR 0.04 [0.02, 0.08] for new construction and 0.15 [0.09, 0.24] for pop-ups, relative to < 400 m.

**Socioeconomic change.** New construction is highest in the lowest-income block groups (Q1 of 2015–19 income) and
in those whose real median income rose most to 2019–23 (top quintile 2.5 vs 0.5 per 1,000 per year in the bottom
quintile); pop-ups rise steadily with baseline income (Q5 1.5 vs Q2 0.4). In the models, one SD of real income
growth goes with IRR 1.35 [1.13, 1.60] (new) and 1.45 [1.24, 1.70] (pop-ups). These are associations: construction
also changes who lives in a block group, and ACS margins of error are large.

## RQ3 — permits and visible construction

| Permits issued 2018–2020 (units) | visible by 2022 | by 2025 |
|---|---|---|
| New construction (2,754 / 2,647) | 58% [57, 60] | 61% [59, 62]; any growth 78% |
| Upward addition, by permit text (497 / 481) | 65% [61, 69] | 67% [62, 71] |
| Any addition / major alteration (9,574 / 9,329) | 7% | 7% (mostly rear or interior work) |

**Time.** For new-construction permits issued 2018–2019, half of the units show new construction within one year
and 70–80% within four; median lag among detected units 1–2 years. 2021 is not observed (no model year), so the curves
are flat there (`figs/rq3_time_to_detection.png`). Units never detected by 2025: 18% (issued 2018) to 39% (2021).

**Without a matching permit.** Share of LiDAR-detected changes with no related permit (building, zoning, demolition,
administrative) on the unit from three years before the period to its end:

| | 2018–2022 | 2022–2025 |
|---|---|---|
| New construction | 3.6% [2.9, 4.5] | 6.8% [5.6, 8.2] |
| Pop-ups | 4.4% [3.5, 5.6] | 9.1% [7.3, 11.3] |
| … with a related permit within 10 m counted | 2.1% / 1.2% | 3.4% / 4.8% |

Most visible change is permitted; the residual is small and partly parcel-matching error (merged parcels, permits
filed on a neighbour). By ZIP: `rq3_unpermitted_by_zip.csv`.

## Floor area and housing (rough)

New floor area = footprint × storeys (median roof height ÷ 3 m); pop-ups = footprint × share of roof that rose ×
added storeys. 122.7 m² of new floor area per home, from 3,307 new-construction permits that state a unit count.

| | covered units | scaled to all units* |
|---|---|---|
| 2018–2022 new construction | 0.81 M m² | 1.28 M m² ≈ 10,400 homes |
| 2018–2022 pop-ups | 0.23 M m² | 0.35 M m² ≤ 2,900 homes |
| 2022–2025 new construction | 0.80 M m² | 0.83 M m² ≈ 6,800 homes |
| 2022–2025 pop-ups | 0.16 M m² | 0.16 M m² ≤ 1,300 homes |

\* Divided by the covered share (64% and 97%); for 2018–2022 this assumes uncovered units behave like covered ones,
which overstates (uncovered units are more peripheral). Pop-up homes are an upper bound: many pop-ups enlarge an
existing home.
