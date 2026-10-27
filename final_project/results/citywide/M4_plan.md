# M4 plan (fixed 2026-10-23, before any citywide data is processed)

M4 is the M3b recipe retrained on the citywide split. It is the only model trained in this phase; nothing is trained after M4.

## Recipe (unchanged from M3b)

- **Model:** LiDAR-anchored bi-temporal unit classifier (`src/unitcls.py`, `scripts/unitcls.py`).
  - Backbone: DINOv3 ViT-L/16 SAT-493M.
  - Heads: per-unit grew and pop-up, focal loss with prior-probability bias, plus auxiliary anchored height regression.
- **Training:** batch 6, bf16. First half with the backbone frozen, then the last 4 blocks unfrozen.
  - Learning rates: head 1e-3 → 2e-4; unit head 3e-4 → 1e-4; backbone 2e-5.
  - Augmentation: flips and colour jitter on the GPU, no rotation. Identity pairs 20%. Changed patches over-sampled.
  - Global image shift as in Stage 3.
- **Periods (years ≤ 2022 only):** 2010→2015, 2010→2018, 2010→2022, 2015→2018, 2015→2022, 2018→2022.
  - **2025 never enters training**, so the 2022→2025 test stays held out in time as well as space.
  - Keeping the recipe unchanged therefore requires the 2010 and 2015 LiDAR for the train and validation areas, in addition to 2018 / 2022 / 2025.
- **Labels:** Stage 1 LiDAR rules (v3 rear filter) per period.
  - "Real-change" masks come from LiDAR-flagged units.
  - Sparse years (2010, 2015) use the `sparse=True` cover rule.
- **Split:** `splits_citywide.md`. Training pairs come from `train` blocks; validation is the `val` blocks, period 2018→2022 only.

## Steps (formula fixed now)

- `iters = 500 × round(8000 × N_train_pairs / 2052 / 500)`, with a minimum of 8,000.
  - N_train_pairs = training pairs over all six periods.
  - 2052 = M3b's training pairs; M3b trained 8,000 steps.
- `frozen_iters = iters / 2`.
- Validation every `max(500, 500 × round(iters / 16 / 500))` steps, i.e. about 16 evaluations.
- Rough expectation: about 5× M3b's data → about 40,000 steps → **7–8 h per run** at 1.5 steps/s.
- **If a run is expected to exceed 5 h, the user is told before it starts** (option: Colab Pro A100).

## Seeds, checkpoints, failures

- **Seeds:** two, 0 and 1.
- **Dead head check:** if validation grew-AP is below 0.15 at step 500, the run is aborted, recorded (`DEAD_<tag>`), and rerun with the next unused seed (2, 3, …).
- **Checkpoints:** each head keeps the step with its own best validation AP, recorded in the dev / final json. Decision thresholds are max-F1 on validation.
- **Resumable:** a checkpoint every 500 steps.
- **Final M4:** the seed with the higher **validation pop-up AP**; tie (difference < 0.005) → higher validation grew AP. Chosen before any test.

## Second-round final test (once, `splits_citywide.md` test ZIPs, 2022→2025)

- **Models:** M3b seed 1 (frozen) and the final M4. Each is run once and logged.
- **Report:**
  - unit-level grew and pop-up AP, recall and precision;
  - 100 m, 250 m and block-group r and total error, with counts calibrated by validation precision ÷ recall;
  - results by area type;
  - positive rates of both periods.
- **Inference model** (section 8 of the task brief): the model with the higher **pop-up correlation at 250 m** on this test. Tie (< 0.01) → block-group pop-up correlation.
- **Pop-ups in intermediate years** are estimated by the model only if that model's pop-up r ≥ 0.8 at 250 m on this test; otherwise only LiDAR anchor years are reported.

## Implementation record (2026-10-26, before any M4 patch is cut or any training)

- **Patch selection** (`scripts/m4_patches.py`): Stage 3 rules on each ZIP's window of the citywide grid (512-cell
  patches, valid share ≥ 0.8 / ≥ 0.3 for 2010–2015, ≥ 50% inside the ZIP). **Added:** ≥ 25% of the patch inside
  the study-area mask, so training stays in the study area's kind of land. Split as in `splits_citywide.md`.
- **Count:** 10,559 training pairs over the six periods (M3b: 2,052); 329 validation pairs (2018→2022).
  8,988 train / val patches to cut (2010: 2,327, 2015: 2,308, 2018: 1,978, 2022: 2,375).
- **Steps from the formula:** iters = 500 × round(8000 × 10,559 / 2,052 / 500) = **41,000**; frozen 20,500;
  validation every 2,500 steps.
- **Expected time:** M3b ran at 1.54 steps/s including validation → about 7.4 h of training per run, plus about 1 h
  of validation (6.7× M3b's validation pairs) → **about 8–8.5 h per run, 16–17 h for both seeds**. Over 5 h, so the
  user is told before training starts.
- **Start condition:** the spot-check decision rule (`spotcheck_plan.md`) — training starts only after the user's
  labels give overall precision ≥ 85%.

## Amendment 1 (2026-10-26, approved by the user before any new training run)

**1. Reason.** The original recipe scaled the steps with the data (41,000 steps, 20,500 frozen). The one-cycle
learning-rate schedule is defined over the steps, so it was stretched by the same factor: warm-up to the peak took
1,025 steps instead of M3b's 200, and the learning rate stayed near the peak for several times longer. Seed 0
(`m4_s0`) collapsed between step 500 and step 2,500: validation grew AP 0.508 at step 500 (the dead-head check
passed), 0.047 at 2,500 and 0.12–0.16 up to 12,500; pop-up AP ≈ 0.005 (chance). Diagnosis on 60 train and 60 val
pairs (`scripts/m4_diagnose.py`, `results/citywide/m4/diagnose_s0.json`): train-pair grew AP 0.656 at step 500 and
0.106 at step 13,500, with the spread of unit logits shrinking from 0.94 to 0.34 — an optimisation collapse, not
over-fitting. Data alignment was checked and is correct. The run was paused at step 13,500; its data, logs,
checkpoints and diagnosis are kept unchanged (`m4_s0` files) and are reported as they are.

**2. Recipe A (replaces "Steps from the formula").** M3b's training schedule exactly: **8,000 steps, the first
4,000 with the backbone frozen**, one-cycle warm-up to the peak in 200 steps and then annealing (5% of each stage,
as in M3b), **validation every 500 steps**. Everything else is the M3b recipe as above; only the training data
change (10,559 citywide train pairs, 329 val pairs). Two seeds. Run tags `m4b_s<seed>`.

**3. Guard.** At **every** validation (not only step 500): if validation grew AP < 0.15, the run stops at once,
is recorded (`DEAD_<tag>` with the step), and the next unused seed starts. Each validation result is checked as soon
as it is written, not at the periodic progress report.

**4. Fallback (fixed now).** If **both seeds 0 and 1** of recipe A trigger the guard **within 2,000 steps**, recipe A
is replaced by a version that resamples to M3b's positive share, everything else unchanged, run with two seeds
(tags `m4c_s<seed>`, same guard):
- Measured now: share of train pairs with ≥ 40 real-change cells (≥ 10 m²) is 70.0% in M3b and 38.8% in M4.
  With the sampling weight 1 + 4 × [n_real ≥ 40] used so far, crops come from such pairs 92.1% of the time in M3b
  and 76.0% in M4.
- The fallback weights those pairs **18.45** instead of 5 (w = 1 + 17.45 × [n_real ≥ 40]), which gives 92.1% for M4.
- No other change is made after this. If the fallback also fails, M4 is reported as failed and M3b seed 1 remains
  the model for the second-round test and inference.

**5. Unchanged.** The second-round final test rules above stay as written; the test ZIPs stay sealed. The final M4
is still the seed with the higher validation pop-up AP (tie < 0.005 → grew AP), chosen before any test.

## Amendment 2 (2026-10-27, approved by the user after recipe A finished; before any test)

**1. Status and reason.** Recipe A finished and M4 was frozen as `m4b_s0` (`m4/m4_selection.md`; validation pop-up AP
0.1233, grew AP 0.720). That selection stays on record and is the **default final M4**. Amendment 2 is decided
*after* seeing these results and is reported as such. Reasons: 4 of 6 recipe-A seeds collapsed within the first 1,500
steps, all while the backbone was still frozen (stage 2 starts at 4,000), so the instability is in the trainable
decoder and unit head, not in backbone fine-tuning; and the pop-up AP (0.12) is low.

**2. Config S (the only new configuration).** Recipe A exactly (M3b schedule: 8,000 steps, 4,000 frozen, then the
last 4 blocks unfrozen; same learning rates and one-cycle schedule; gradient clipping at norm 1.0 as before;
validation every 500 steps on the same 329 validation pairs), with four changes aimed at the observed failure:
- sampling weight of pairs with ≥ 40 real-change cells **18.45** instead of 5 (the value pre-computed in amendment 1
  to restore M3b's 92.1% positive share);
- batch **16** instead of 6;
- **EMA** of the weights, decay 0.999 with warm-up min(0.999, (1+t)/(10+t)); validation, the guard and the saved
  checkpoints use the EMA weights;
- unit pooling **mean + max** (`--pool meanmax`): adds the max of each feature, the max predicted rise and the share of
  cells predicted to rise ≥ 2.5 m, because a rear pop-up covers only part of a unit and the LiDAR rule itself uses
  the share of roof that rose.
Not done: a new backbone branch (config L), ensembles, test-time augmentation, per-period thresholds.
Run tags `m4s_s<seed>`; trained on the AutoDL cloud GPU (RTX 4080 SUPER), recipe A was trained locally (RTX 5060 Ti):
the hardware differs and is reported.

**3. Guard and seeds.** Unchanged guard: at every validation, grew AP < 0.15 → stop, `DEAD_<tag>`, next seed. Seeds
0, 1, 2, … until two seeds finish (at most seed 5); seeds 0 and 1 may run at the same time on the one GPU, and the
first two finished seeds in seed order are used. Fewer than two finished seeds in 0–5 → config S failed, M4 stays
`m4b_s0`.

**4. Decision rule (fixed now, `scripts/m4_compare.py`).** On the validation units, each run scored at its best
pop-up step (the same rule as for `m4b_s0`): statistic = mean validation pop-up AP of the two S seeds minus that of
`m4b_s0`; 95% interval from 2,000 bootstrap resamples of the validation 1 km blocks (seed 0). **Config S replaces
`m4b_s0` only if the lower end of the interval is above 0.** Then the final M4 is the S seed with the higher
validation pop-up AP (tie < 0.005 → higher grew AP). Otherwise the final M4 stays `m4b_s0`. The result is written
to `m4/m4_compare.json` and the selection record is updated before any test.

**5. Unchanged.** The second-round test rules, the sealed test ZIPs and everything after M4 stay as written.

## Round-2 test: implementation record (2026-10-27, written before any test output is seen)

The test section above is unchanged; this records how it is carried out (`scripts/m4_test.py`).
- **Data (local CPU):** test-ZIP rasters, units and LiDAR labels for 2018→2022 and 2022→2025 with
  `scripts/citywide_change.py --role test` (same rules as train / val; prints nothing about the test ZIPs). High-rise
  flags with `scripts/highrise_flag.py --role all` (train / val and test ZIPs together, so cells and block groups
  that straddle a test ZIP use both parts, as `study_area.md` requires).
- **Patches:** 512-cell patches of each test ZIP's window with LiDAR height on ≥ 80% of cells, ≥ 50% inside the
  ZIP and ≥ 25% inside the study-area mask (the train / val rules, without a split), years 2022 and 2025; pairs =
  patches present and not blank in both years. Units scored exactly as in validation (256-cell windows, stride 128,
  unit ≥ 90% inside a window, LiDAR coverage in both years; mean logit).
- **Models:** final M4 `m4s_s0` (pop-up head `unitcls_m4s_s0.pt`, grew head `unitcls_m4s_s0_grew.pt`, step 8,000)
  and M3b seed 1 `v3b_s1` (frozen). Each uses its own validation thresholds and its own count calibration
  (precision ÷ recall at those thresholds on its own validation units: citywide validation for M4, Stage 3
  validation for M3b). Nothing is tuned on the test.
- **High-rise:** flagged 100 m / 250 m cells and block groups are left out of the main grid results and reported
  separately; at unit level, units in a flagged 100 m cell are left out of the main results and reported separately.
- **Reported:** unit-level grew / pop-up AP, precision and recall (with intervals); calibrated counts per 100 m,
  250 m cell and block group (r, MAE, mean error, totals) for grew, new and pop-up; unit-level results by the unit's
  own area type and by ZIP type; LiDAR positive rates of 2018→2022 and 2022→2025 in the test ZIPs.
- **Inference model:** higher calibrated pop-up r at 250 m (main cells); tie < 0.01 → block-group pop-up r.
  Intermediate-year pop-ups are estimated only if that r ≥ 0.8.
- **Once:** `results/citywide/m4/round2_test_log.csv` is written before any scoring; a second run is refused.
- **Where:** everything local (test data never leave the local machine); the M4 checkpoints were copied from the
  cloud (sha256 checked). Model scoring on the local GPU (inference only).
