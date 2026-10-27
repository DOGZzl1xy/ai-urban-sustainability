"""M4 amendment 2 decision: config S (two live seeds) vs m4b_s0 on the validation units, block bootstrap.

    POPUP_DATA=citywide python scripts/m4_compare.py --base m4b_s0 --cand m4s_s0 m4s_s1 [--n 2000]

Inputs: data/interim/unitcls_preds/{tag}_valbest_2018_2022.parquet (validation units scored at each run's best
pop-up step; written by unitcls.py train, or by unitcls.py val for older checkpoints). All runs must score the same
units. Statistic: mean pop-up AP of the candidates minus the base's pop-up AP. Resampling unit: the 1 km block of
the first patch that scored the unit (the split's own block grid), drawn with replacement, 2,000 times, seed 0.
Rule (fixed before training, M4_plan.md amendment 2): config S is adopted only if the lower end of the 95%
interval is above 0; then the final M4 is the candidate with the higher validation pop-up AP (tie < 0.005 -> higher
validation grew AP). Otherwise the final M4 stays the base. Output: results/citywide/m4/m4_compare.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import height_data as H  # noqa: E402
from rasters import FT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="m4b_s0")
ap.add_argument("--cand", nargs=2, required=True)
ap.add_argument("--n", type=int, default=2000)
a = ap.parse_args()
PRED = ROOT / "data/interim/unitcls_preds"


def ap_score(y, s):
    o = np.argsort(-s, kind="stable")
    y = np.asarray(y)[o]
    tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


tags = [a.base, *a.cand]
assert len(set(tags)) == 3, "base and the two candidates must be different runs"
runs = {t: pd.read_parquet(PRED / f"{t}_valbest_2018_2022.parquet") for t in tags}
key = ["zip", "unit"]
df = runs[a.base][key + ["i", "j", "popup", "grew"]].copy()
for t in tags:
    r = runs[t]
    assert len(r) == len(df) and set(map(tuple, r[key].to_numpy())) == set(map(tuple, df[key].to_numpy())), \
        f"{t} scored different validation units"
    df = df.merge(r[key + ["logit_popup", "logit_grew"]].rename(
        columns={"logit_popup": f"p_{t}", "logit_grew": f"g_{t}"}), on=key)

# 1 km block of the first patch that scored each unit (patch centre, as in cloud/pack.py)
pt = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})
pt = pt[pt.year == 2018][["zip", "i", "j", "x0", "y1"]]
df = df.merge(pt, on=["zip", "i", "j"], how="left")
assert df.x0.notna().all(), "patch without coordinates"
blk, half = 1000 / FT, H.P * H.CELL / 2
df["block"] = [f"{int((x + half) // blk)}_{int((y - half) // blk)}" for x, y in zip(df.x0, df.y1)]

y, yg = df.popup.to_numpy(), df.grew.to_numpy()
point = {t: {"ap_popup": ap_score(y, df[f"p_{t}"].to_numpy()), "ap_grew": ap_score(yg, df[f"g_{t}"].to_numpy()),
             "iter": int(runs[t]["iter"].iloc[0])} for t in tags}
diff = np.mean([point[t]["ap_popup"] for t in a.cand]) - point[a.base]["ap_popup"]

rng = np.random.default_rng(0)
groups = df.groupby("block").indices
blocks = list(groups)
boot = []
for _ in range(a.n):
    idx = np.concatenate([groups[blocks[k]] for k in rng.integers(0, len(blocks), len(blocks))])
    yy = y[idx]
    if yy.sum() == 0:
        continue
    c = np.mean([ap_score(yy, df[f"p_{t}"].to_numpy()[idx]) for t in a.cand])
    boot.append(c - ap_score(yy, df[f"p_{a.base}"].to_numpy()[idx]))
lo, hi = np.percentile(boot, [2.5, 97.5])
adopt = bool(lo > 0)
if adopt:
    c0, c1 = a.cand
    d = point[c0]["ap_popup"] - point[c1]["ap_popup"]
    final = (c0 if d > 0 else c1) if abs(d) >= 0.005 else \
        (c0 if point[c0]["ap_grew"] >= point[c1]["ap_grew"] else c1)
else:
    final = a.base
out = {"base": a.base, "candidates": a.cand, "val_units": int(len(df)), "val_popups": int(y.sum()),
       "blocks": len(blocks), "point": point, "diff_mean_cand_minus_base": diff,
       "ci95": [float(lo), float(hi)], "bootstrap_n": len(boot), "adopt_config_S": adopt, "final_m4": final}
res = ROOT / "results/citywide/m4/m4_compare.json"
res.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
