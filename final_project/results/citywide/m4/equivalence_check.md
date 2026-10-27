# Equivalence check: saved masks / unit ids (POPUP_PREP_CACHE=1) vs rebuilding from the rasters

Run 2026-10-26 21:05 by `scripts/prep_cache.py check` (seed 20261026).

- Sample: 50 random train / val pairs over all six periods, from ZIP windows up to 40 M cells (rebuilding a larger window needs several GB of memory while training runs); 12 ZIPs, 6 periods, 9 val pairs, 34 pairs with real change.
- Compared exactly (every cell): the real-change mask of the pair, the unit-id patch, and the cells per unit over the whole ZIP grid (used to decide whether a unit lies >= 90% inside a crop).

**Result: PASS - all identical**

| | identical |
|---|---|
| masks | 50 / 50 |
| unit-id patches | 50 / 50 |
| unit sizes (per ZIP) | 12 / 12 ZIPs |

Per pair:

|   zip | period    |   i |   j | split   |   mask_cells | mask_identical   | uid_identical   |
|------:|:----------|----:|----:|:--------|-------------:|:-----------------|:----------------|
| 19122 | 2010-2015 |   5 |   1 | train   |            0 | True             | True            |
| 19122 | 2010-2022 |   1 |   6 | train   |         1765 | True             | True            |
| 19122 | 2015-2022 |   4 |   5 | train   |         5426 | True             | True            |
| 19123 | 2010-2015 |   2 |   6 | val     |         1722 | True             | True            |
| 19123 | 2010-2018 |   1 |   7 | val     |        30384 | True             | True            |
| 19123 | 2010-2022 |   1 |   5 | val     |        10357 | True             | True            |
| 19123 | 2010-2022 |   2 |   7 | val     |        11725 | True             | True            |
| 19123 | 2010-2022 |   4 |   7 | val     |         8453 | True             | True            |
| 19123 | 2015-2018 |   1 |   4 | val     |         1286 | True             | True            |
| 19123 | 2015-2022 |   1 |   2 | train   |            0 | True             | True            |
| 19125 | 2010-2015 |   9 |   0 | train   |         1162 | True             | True            |
| 19125 | 2010-2022 |   6 |   2 | train   |         2320 | True             | True            |
| 19125 | 2010-2022 |   6 |   3 | train   |         3363 | True             | True            |
| 19125 | 2015-2018 |   9 |   5 | train   |            0 | True             | True            |
| 19126 | 2010-2022 |   5 |   2 | train   |            0 | True             | True            |
| 19126 | 2018-2022 |   2 |   3 | train   |            0 | True             | True            |
| 19129 | 2010-2022 |   3 |   9 | train   |            0 | True             | True            |
| 19129 | 2015-2018 |   9 |  11 | train   |            0 | True             | True            |
| 19129 | 2018-2022 |   7 |  12 | train   |         8275 | True             | True            |
| 19132 | 2010-2022 |   6 |   7 | train   |         3656 | True             | True            |
| 19132 | 2015-2018 |   4 |  12 | train   |            0 | True             | True            |
| 19132 | 2015-2018 |   8 |  12 | train   |          528 | True             | True            |
| 19132 | 2015-2022 |   6 |   7 | train   |         2310 | True             | True            |
| 19133 | 2010-2015 |   3 |   3 | train   |          603 | True             | True            |
| 19133 | 2010-2015 |   4 |   4 | train   |          178 | True             | True            |
| 19133 | 2010-2018 |   1 |   2 | train   |         3567 | True             | True            |
| 19133 | 2010-2018 |   5 |   5 | train   |          513 | True             | True            |
| 19133 | 2015-2018 |   3 |   1 | train   |          381 | True             | True            |
| 19133 | 2015-2022 |   5 |   1 | train   |          655 | True             | True            |
| 19133 | 2015-2022 |   6 |   1 | train   |         3430 | True             | True            |
| 19138 | 2015-2022 |   6 |   6 | train   |            0 | True             | True            |
| 19141 | 2010-2022 |   4 |   5 | train   |           36 | True             | True            |
| 19141 | 2010-2022 |   6 |   4 | train   |            0 | True             | True            |
| 19141 | 2010-2022 |   9 |   7 | train   |            5 | True             | True            |
| 19141 | 2015-2018 |   3 |   5 | train   |           29 | True             | True            |
| 19141 | 2015-2018 |   6 |   4 | train   |            0 | True             | True            |
| 19141 | 2018-2022 |   4 |   7 | train   |            0 | True             | True            |
| 19141 | 2018-2022 |   6 |   7 | train   |            0 | True             | True            |
| 19141 | 2018-2022 |  10 |   3 | train   |            7 | True             | True            |
| 19142 | 2010-2018 |   1 |   4 | train   |            0 | True             | True            |
| 19142 | 2018-2022 |   5 |   8 | train   |          163 | True             | True            |
| 19142 | 2018-2022 |   6 |   6 | train   |            0 | True             | True            |
| 19146 | 2010-2018 |   3 |  12 | train   |         4037 | True             | True            |
| 19146 | 2015-2018 |   1 |   7 | train   |            0 | True             | True            |
| 19146 | 2015-2018 |   7 |  11 | train   |         1321 | True             | True            |
| 19146 | 2018-2022 |   2 |   8 | train   |         1379 | True             | True            |
| 19146 | 2018-2022 |   6 |  10 | train   |         4832 | True             | True            |
| 19147 | 2015-2018 |   1 |   2 | val     |         2369 | True             | True            |
| 19147 | 2015-2018 |   3 |   2 | val     |          544 | True             | True            |
| 19147 | 2018-2022 |   4 |   7 | val     |         2570 | True             | True            |
