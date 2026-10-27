# LiDAR rule spot-check in new areas (section 七) — plan, written before any label

The change rules (new construction, pop-up) were checked only in ZIP 19146 (recall 96.6% / 87.1%, fresh-sample
precision 95.5%). Before the citywide labels train M4, the user checks a fixed random sample of detections in areas
the rules have never been checked in.

## Sample (drawn once, `scripts/spotcheck.py`, seed 20261026)

- **Pool:** train / val ZIPs that were **not** used in Stage 3: not the 7 Stage 3 training ZIPs, not 19146 / 19147,
  and no second-round test ZIP. Units inside the study-area mask with full LiDAR coverage in both years.
- **Detections:** units flagged new construction or pop-up (the two classes M4 learns).
- **Periods:** 2018→2022 and 2022→2025 (the two anchor periods).
- **Per area type** (rowhouse, detached, mixed): 20 detections, 10 per period. If a period has fewer than 10, the
  other period fills up. If the area type has fewer than 20 in total, all of them are shown.
- The item list (`spotcheck_items.csv`: item, ZIP, unit id, period, area type) is committed. The review page
  `review_spotcheck.html` shows addresses and stays local.

## Labels

Per case: real change / no change / unsure, with a note for "no change". The page shows the two aerial images and
the LiDAR height change. Export `review_spotcheck_labels.csv` next to the page.

## Decision rule (fixed now)

- Precision = real / (real + no change), with a bootstrap 95% CI, overall and per area type and per class.
- **Overall precision ≥ 85%:** the rules are used citywide unchanged.
- **An area type below 75%:** its labels are reported as less reliable, and the round-2 results for that area type
  carry that note. The rules are not tuned on this sample (it is too small to tune on and would then no longer be a
  fair check). Any rule change would need a new, separate sample and is decided with the user.
- **Overall below 85%:** M4 training waits; the false positives are categorised and discussed with the user first.

## Amendment (2026-10-26, before any label)

The first draw was deleted unseen: the 2025 vegetation fix (`veg2025_fix.md`) changes the 2022→2025 detection pool.
The sample is redrawn after the rebuild with the same design and seed.
