# Stage 6: study-area estimates 2018–2025 (fixed 2026-10-28, after the round-2 test, before any stage-6 output)

Goal: for every year 2018–2025, how much construction happened where, at 100 m, 250 m and 2019 block groups, for the
whole study area (46 ZIPs, test ZIPs included now that M4 is final and tested). This is the data for stage 7.

## 1. LiDAR anchor periods (measured; local)

2018→2022 and 2022→2025, per unit, from `data/interim/lidar_change_city/` (Stage 1 v3 rules: new construction,
pop-up, demolition; already built for all ZIPs). Units counted: inside the study-area mask and covered by LiDAR in
both years (the rule used everywhere). Counted per 100 m / 250 m cell (unit centroid) and block group. Pop-ups are
reported **only** for these two periods.

## 2. Intermediate years (model; cloud GPU)

- **Pairs:** 2018 anchor (LiDAR + image) with the 2019 and 2020 images; 2022 anchor with the 2023 and 2024 images.
  Each pair estimates change since its anchor year.
- **Model:** final M4 `m4s_s0` (grew and pop-up heads), its validation thresholds. **Outputs used: new construction
  (grew and not pop-up) as the main series, grew as a secondary series.** Pop-ups are not estimated: the round-2
  pop-up r at 250 m was 0.42 < 0.8 (M4_plan.md rule).
- **Patches and units:** the round-2 rules on every ZIP window: 512-cell patches with LiDAR height on ≥ 80% of cells
  in the anchor year, ≥ 50% inside the ZIP, ≥ 25% inside the mask; units scored as in validation (256-cell windows,
  stride 128, unit ≥ 90% inside a window). The target year has no LiDAR, so only the anchor-year coverage applies.
  The 2018 flight is sparse outside the centre, so 2019 / 2020 cover fewer units than 2023 / 2024; the share of
  units scored is reported per cell and year.
- **Counts:** flags × validation precision ÷ recall (factor 0.90 new, 0.92 grew), the calibration tested in round 2
  (test: new 85 estimated vs 97 measured; grew 221 vs 143 — grew is over-estimated through pop-up false positives,
  which is why new construction is the main series). A correction by the validation false-positive rate was checked
  on the test units and is worse (new 4 vs 97), so it is not used.
- **Where:** patches are cut and scored on the cloud (batch 1 + batch 2 data); only per-unit scores (unit ids, no
  addresses) come back. Calibration, aggregation and maps are done locally.

## 3. Yearly series and checks

- Cumulative change since the anchor: 2019 and 2020 from the model, 2022 and 2025 from LiDAR (2018→2022 and
  2022→2025), 2023 and 2024 from the model on the 2022 anchor.
- Consistency checks, reported and not used to adjust anything: per cell, model 2018→2019 ≤ 2018→2020 ≤ LiDAR
  2018→2022 (and the same for 2022 → 2023 / 2024 / 2025 LiDAR), in counts after calibration.
- High-rise cells and block groups (`highrise_all.csv`) are kept out of the main results and reported separately.
- Model-based years are labelled as estimates with the round-2 test errors next to them.

## Decision after the stage-6 run (2026-10-28, approved by the user)

Intermediate years report **new construction only**. The grew series over-shoots its own LiDAR end point (2022
anchor: model 2024 = 2,144 vs LiDAR 2025 = 1,837 at 250 m, main cells), as the round-2 test predicted (grew +55%);
its numbers stay in `results/citywide/stage6/` as an appendix and are not used in the main results.
