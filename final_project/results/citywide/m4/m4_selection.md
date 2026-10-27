# M4 selection (recipe A, amendment 1) — frozen 2026-10-27 00:53, before any test

Rule (`M4_plan.md`, written before training): the final M4 is the seed with the higher **validation pop-up AP**;
difference < 0.005 → higher validation grew AP. A run whose validation grew AP falls below 0.15 at any validation is
stopped (`DEAD_<tag>`) and the next unused seed is run, until two seeds finish.

Local queue `queue_a1.sh` (local GPU, 34 GB RAM), 2026-10-26 15:15 → 2026-10-27 00:53. Validation: 329 pairs
(2018-2022), 29,353 units, 117 pop-ups.

| run | result | val AP grew by step (500, 1000, …) | best val pop-up AP (step) | best val grew AP |
|---|---|---|---|---|
| `m4b_s0` | finished 8,000 | 0.644 at 4,000 → 0.703 at 7,500 | **0.1233** (7,500) | 0.720 |
| `m4b_s1` | stopped at 1,000 | 0.247, 0.099 | — | — |
| `m4b_s2` | stopped at 500 | 0.026 | — | — |
| `m4b_s3` | stopped at 500 | 0.010 | — | — |
| `m4b_s4` | stopped at 1,500 | 0.415, 0.397, 0.010 | — | — |
| `m4b_s5` | finished 8,000 | 0.385, 0.386, 0.445, 0.518, … 0.583 at 6,500, 0.530 at 8,000 | 0.116 (8,000) | 0.583 |

**Final M4 = `m4b_s0`** (`data/interim/models/unitcls_m4b_s0.pt`, pop-up head at step 7,500; grew head
`unitcls_m4b_s0_grew.pt`). Pop-up AP 0.1233 vs 0.116: difference 0.0073 ≥ 0.005, so the tie rule does not apply.
Thresholds: `unitcls_m4b_s0_val.json` / `_val_grew.json`.

The fallback (`m4c`, pos-weight 18.45) was not triggered: it applies only if seeds 0 and 1 both stop within 2,000 steps,
and seed 0 finished.

For the report: 4 of 6 recipe-A seeds collapsed in the first 1,500 steps (grew AP falling to chance after a normal
start in s1 and s4), so recipe A on the citywide data is unstable early in training; the two surviving seeds differ
a lot in grew AP (0.720 vs 0.583). Memory warnings (≤ 0.2 GB free) occurred in s0 (141, during packing), s4 and s5,
but not in s2 or s3, which collapsed fastest, so memory pressure is not the cause.

## Amendment 2 (config S) — decided 2026-10-27 19:20 CST (07:20 EDT), before any test

Config S (`M4_plan.md` amendment 2: recipe A + sampling weight 18.45, batch 16, EMA 0.999, mean+max unit pooling),
queue `queue_m4s.sh` on the AutoDL cloud GPU (RTX 4080 SUPER), 2026-10-27 14:05 → 19:18 CST. Same 329 validation
pairs, 29,353 units, 117 pop-ups. Neither seed triggered the guard.

| run | result | val AP grew by step (500, 1000, …) | best val pop-up AP (step) | best val grew AP |
|---|---|---|---|---|
| `m4s_s0` | finished 8,000 | 0.589, 0.661, 0.721, 0.745 … 0.799 at 8,000 | **0.4261** (8,000) | 0.799 |
| `m4s_s1` | finished 8,000 | 0.662, 0.720, 0.736, 0.756 … 0.804 at 8,000 | 0.3325 (8,000) | 0.804 |

Decision (`scripts/m4_compare.py`, `m4_compare.json`): all three runs scored on the cloud GPU at their best pop-up
step. `m4b_s0` re-scored there gives pop-up AP 0.1310 / grew AP 0.7049 (0.1233 / 0.7027 on the local GPU from the
same checkpoint; same units, so the difference is GPU arithmetic in bf16 — recorded, and it makes the comparison
harder for config S). Mean S pop-up AP minus `m4b_s0`: **+0.248, 95% block-bootstrap interval [0.153, 0.314]**
(2,000 resamples of 42 validation 1 km blocks). The lower end is above 0 → **config S is adopted**.

**Final M4 = `m4s_s0`** (pop-up AP 0.4261 vs 0.3325 for `m4s_s1`, difference ≥ 0.005). Checkpoints on the cloud:
`data/interim/models/unitcls_m4s_s0.pt` (pop-up head) and `unitcls_m4s_s0_grew.pt` (grew head; both step 8,000),
backed up to `/root/autodl-fs/m4_final/` with SHA256SUMS. Validation thresholds (max F1, logits): grew 0.142
(precision 0.744, recall 0.809), pop-up −0.099 (precision 0.350, recall 0.530) — `unitcls_m4s_s0_val*.json`.
This replaces the amendment-1 selection above (`m4b_s0`), which stays on record. M4 is now final; nothing more is
trained.

For the report: the change is post hoc (decided after recipe A) and the hardware differs from recipe A (cloud vs
local GPU). The two S seeds differ in pop-up AP (0.43 vs 0.33), so seed variance is large; validation has only 117
pop-ups.
