# Stage 3: detecting vertical change from aerial imagery

Status: **done** (2026-10-23 04:15). Final model: **M3b, seed 1**. Test protocol:
- **Development test:** 2018 → 2022 in the test ZIPs 19146 + 19147. Open to every model version.
- **Final test:** 2022 → 2025 in the same ZIPs. **Sealed:** run once, for the one final model (`final_test_log.csv`).

## Model versions

These names replace the earlier "v1/v2/v3" labels, which clashed with the rear-addition filter versions of the
LiDAR rules.

| Name | Model | Input | Training data | Output |
|---|---|---|---|---|
| **M0** | single-image DPT (Stage 2 pilot) | one image | 2018 imagery + LiDAR, 4 training ZIPs | height per pixel (+ vegetation) |
| **M1** | version A: two images | image t0 + image t1 | 2018→2022, 4 training ZIPs | height change per pixel |
| **M2** | version B: LiDAR-anchored | LiDAR height t0 + image t0 + image t1 | 2018→2022, 4 training ZIPs | height at t1 per pixel |
| **M3** | v3r: unit classifier on M2 | as M2 + parcel units | 2018→2022, 7 training ZIPs | per unit: grew, pop-up (+ pixel height, auxiliary) |
| **M3b** | v3b: M3 with older years | as M3 | 6 periods (2010→2015 … 2018→2022), 7 training ZIPs | as M3 |

All use DINOv3 ViT-L/16 SAT-493M except M0's baselines. Validation is always the Stage 3 validation blocks
(training ZIPs, 2018→2022, stratified by pop-ups: 6,637 units, 88 pop-ups, 241 grew).

## Development-test results (2018 → 2022, test ZIPs)

| Model | New construction | Pop-ups | Notes |
|---|---|---|---|
| M0, both years predicted (pilot rules) | 19146: R 92% / P 91% | 19146: R 44% / P 44% | Stage 2 pilot |
| M0, LiDAR before + predicted after (thresholds tuned on validation) | 19146: R 99% / P 94%; 19147: R 97% / P 77% | 19146: R 41% / P 50%; 19147: R 17% / P 13% | tuning blocks partly in M0's training data → optimistic |
| M1 | not evaluated | not evaluated | training collapsed: validation change error equal to "no change everywhere" at every check; stopped at step ~2,500 |
| M2 | 19146: R 0%; 19147: R 2% | R 0% in both; predicted rise 0.1 m vs 3.8 m | copies the anchor (building MAE 0.32 / 0.26 m) |
| M3, first run (before the stability fix) | grew AP 0.750 (R 61% / P 81%) | AP 0.228 (R 56% / P 24%) | 1 run; 3 of 4 later runs had a dead unit head → not reproducible |
| M3 (v3r), seed 0 | grew AP 0.810 (R 69% / P 83%) | AP 0.184 (R 60% / P 25%) | old data path; checkpoint step 5,500 for both heads |
| M3 (v3r), seed 3 | grew AP 0.735 (R 64% / P 77%) | AP 0.288 (R 45% / P 34%) | seed 1 had a dead head at step 500 → replaced by seed 3; steps 4,000 (grew) / 4,500 (pop-up) |
| M3b (v3b), seed 0 | grew AP 0.761 (R 62% / P 82%) | AP 0.186 (R 59% / P 21%) | steps 7,000 (grew) / 6,000 (pop-up) |
| M3b (v3b), seed 1 | grew AP 0.790 (R 64% / P 84%) | AP 0.437 (R 42% / P 60%) | step 6,000 for both heads |

AP = average precision of the unit scores. The base rates in the test ZIPs are 3.2% (grew) and 1.6% (pop-ups).

### Two-seed comparison: do the older years help? (development test, both test ZIPs)

| Configuration | Grew AP: mean (range) | Pop-up AP: mean (range) | Pop-up grid r, 250 m: mean | Pop-up grid r, block group: mean |
|---|---|---|---|---|
| M3 (2018→2022 only) | 0.773 (0.735–0.810) | 0.236 (0.184–0.288) | 0.70 | 0.70 |
| M3b (+2010/2015 periods) | 0.775 (0.761–0.790) | 0.311 (0.186–0.437) | 0.79 | 0.77 |

- **Grew:** the older years make no difference.
- **Pop-ups:** M3b is better on average (+0.075 AP), but the difference is smaller than the spread between seeds (0.186 vs 0.437 within M3b). So there is weak evidence that the older years help, and no firm conclusion.
- **New construction** is derived as grew and not pop-up. When the pop-up head over-flags (M3 seed 0, M3b seed 0: about 1,100–1,300 pop-up flags for 468 true ones), new-construction counts collapse: grid r 0.01–0.5. With a good pop-up head it is 0.92–0.99.

Grid-level evaluation (`grid_eval_<tag>.csv`, `figs/grid_eval_<tag>.png`):
- Calibrated counts are flags × validation precision ÷ validation recall.
- Correlation for grew: 0.90–0.92 at 100 m, 0.95–0.96 at 250 m, 0.96–0.97 at block group, in all four runs.
- Pop-ups, best run (M3b seed 1): 0.71 / 0.87 / 0.84.

False pop-ups on the development test, by what LiDAR shows (`fp_classes_<tag>.csv`):

| Run | False pop-ups | (a) new construction | (b) near threshold | (c) rear rise | (d) no change | (e) other |
|---|---|---|---|---|---|---|
| M3 seed 0 | 820 | 426 | 109 | 40 | 192 | 53 |
| M3 seed 3 | 407 | 122 | 74 | 14 | 176 | 21 |
| M3b seed 0 | 1,020 | 413 | 138 | 34 | 364 | 71 |
| M3b seed 1 | 135 | 17 | 48 | 4 | 53 | 13 |

Most errors of the weaker runs are real changes of another kind: (a) new construction, (b) near-threshold rises, (c) rear rises. Category (d) in M3b seed 1 has 53 units; a review page with 30 random ones is in `review_fp_d_v3b_s1.html` (local only: it links addresses).

## Decision (criteria fixed before the runs)

- **Criteria:** pop-up AP ≥ 0.35, or pop-up grid-level r ≥ 0.8.
- **Configuration:** chosen by the two-seed mean, M3b.
  - Its mean pop-up AP is 0.311 and its mean grid r is 0.79 (250 m) / 0.77 (block group): **both just below the bar**.
  - The best single run (M3b seed 1: 0.437, 0.87 / 0.84) passes both, but it is not reproducible across seeds.
- **Outcome:** model iteration stops.
  - The final model covers grew and new construction.
  - Pop-ups are reported as a limitation.
- **Final model:** M3b; seed 1 chosen by validation AP (pop-up 0.336 vs 0.113, grew 0.838 vs 0.792). Nothing about 2022→2025 was used.

## Final test, 2022 → 2025 (held out in time and space), run once

- Model: M3b seed 1, checkpoint step 6,000 for both heads. Thresholds were fixed on validation: grew −0.050, pop-up −0.956 (logits).
- Units fully covered in both years: 27,996.

| Test set | Grew: R / P / AP | Pop-up: R / P / AP |
|---|---|---|
| Both ZIPs | 50% [45, 55] / 58% [53, 64] / **0.561** | 33% [26, 39] / 27% [21, 32] / **0.159** |
| 19146 | 54% / 65% / 0.631 | 37% / 30% / 0.186 |
| 19147 | 33% / 36% / 0.336 | 23% / 19% / 0.115 |

Grid level, calibrated counts vs LiDAR (`grid_eval_v3b_s1_FINAL.csv`):

| Class | r at 100 m | r at 250 m | r at block group | Estimated total vs LiDAR |
|---|---|---|---|---|
| Grew | 0.72 | 0.83 | 0.88 | 345 vs 369 |
| New construction | 0.76 | 0.87 | 0.88 | 175 vs 163 |
| Pop-up | 0.44 | 0.61 | 0.74 | 367 vs 206 (overestimated) |

**Reading:**
- Moving to an unseen year costs a lot at unit level: grew AP 0.79 → 0.56, pop-up AP 0.44 → 0.16. The 2022→2025 pop-up result is close to the development-test results of the weaker seeds, so seed 1's development-test pop-up score was partly luck.
- **Aggregated to 250 m cells or block groups**, the model still tracks where growth and new construction happen (r 0.83–0.88, totals within 10%). That is the scale of the urban analysis (RQ2).
- Likely reasons for the drop:
  - 2025 imagery is a year the model never saw.
  - 2025 LiDAR is sparser (≈30 vs 43 pts/m²), so its change labels are noisier.
  - Image–LiDAR registration scatters more in 2025.
  - 2022→2025 is a 3-year window with fewer changes (369 grew vs 915 in 2018→2022).
- The first attempt at this test crashed before producing any prediction (a missing unit-id file for a 2022-only patch). It is recorded in `final_test_attempt1_aborted.*`; after the fix the test ran once (`final_test_log.csv`).

## Run times (overnight 2026-10-22/23, no errors, no memory warnings, no interruptions)

| Run | Data path | Steps | Time | Steps / s incl. validation |
|---|---|---|---|---|
| v3r seed 0 | old | 6,000 | 80 min | 1.25 |
| v3b seed 0 | new | 8,000 | 88 min | 1.52 |
| v3r seed 1 | new | 500 | 7 min | aborted: dead head (grew AP 0.025) |
| v3r seed 3 | new | 6,000 | 66 min | 1.52 |
| v3b seed 1 | new | 8,000 | 87 min | 1.53 |

## Notes on the runs

- **Dead unit head:**
  - In 3 of 4 runs of the first unit classifier, the head stayed at the base rate.
  - Cause: with p = 0.5 at the start, the rare-positive focal loss pushed every logit far negative and stopped learning.
  - Fix: a prior-probability bias, plus a lower learning rate for the unit head.
  - It still happened once after the fix, so every run is now checked at step 500. If validation grew-AP is below 0.15, the run is aborted and retried with the next seed. Live runs are at 0.59–0.73 at that point.
- **Checkpoint selection:** each head (grew, pop-up) uses the checkpoint with its own best validation AP, and the step is recorded in `unitcls_<tag>_dev.json` → `checkpoints`. **v3r seed 0** was trained before both were saved, so its grew head uses the pop-up-best checkpoint.
- **Data loading:**
  - **v3r seed 0** used the old path: single process, compressed `.npz`, CPU augmentation.
  - The other three runs use the new path: memory-mapped `.npy`, 6 DataLoader workers, flips and colour jitter on the GPU (rotations off), cudnn.benchmark, TF32 and channels_last.
  - **Speed check** (`smoke_speed.log`): 200 steps each, same configuration, seed 0, steady-state rate from the per-step log, GPU utilisation sampled every second.

    | Data path | Steps / s | GPU util. (median) | Loss mean, steps 10–100 / 110–200 | Max loss |
    |---|---|---|---|---|
    | old | 1.42 | 51% | 0.89 / 1.17 | 2.97 |
    | new, with 90° rotations | 2.03 | 68% | 1.12 / 1.05 | 3.27 |
    | **new, flips only (used)** | **1.97 (+39%)** | **72%** | 1.03 / 0.90 | 2.96 |

  - The loss curves are of the same size and noise. 90° rotations are available (`--rotate`) but **off**, so the augmentation matches v3r seed 0 and the two seeds stay comparable.
  - What still limits GPU use is the per-sample loops: unit targets on the CPU, flips on the GPU.

## Files

- `unitcls_<tag>_{train.csv, val.json, val_grew.json, dev.json}`
- per-unit predictions in `data/interim/unitcls_preds/` (unit ids only)
- `grid_eval_<tag>.*`, `grid_eval_v3b_s1_FINAL.*`, `fp_classes_<tag>.csv`, `unitcls_v3b_s1_FINAL.json`, `final_test_log.csv`

## Note added 2026-10-26: 2025 vegetation

The 2025 flight (18 April) has 42–58% of its high-vegetation points as single returns, so the return-structure
vegetation rule let some canopy count as building in 2025 (`results/citywide/veg2025_fix.md`). The round-1 final test
above used those 2025 labels. With the corrected rule (classes 3/4/5 as vegetation), the 19146 / 19147 truth would
hold slightly fewer detections (new construction 179 → 172, pop-ups 237 → 230 on fully covered units inside the citywide study area). The test is
not rerun; its numbers stand as reported, with this note.
