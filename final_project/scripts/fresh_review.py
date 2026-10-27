"""Fresh precision check for the v3 rules: 50 random v3-flagged units not in the first review.

v2/v3 were designed after looking at the first 100 labels, so their precision there is optimistic. This sample is
drawn once, with a new seed, from v3 flags excluding every unit already reviewed.

    python scripts/fresh_review.py          # writes review_v3_fresh_items.csv + review_v3_fresh.html (local only)
    python scripts/fresh_review.py score    # reads review_v3_fresh_labels.csv, writes review_v3_fresh_scores.json
"""

import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import change as C  # noqa: E402
import rear_filter as RF  # noqa: E402
from match import bootstrap_rate  # noqa: E402

OUT = ROOT / "results/popup_feasibility"
SEED = 20261022
N = 50


def v3_flags():
    f = RF.apply_v2(C.apply_rules(pd.read_parquet(ROOT / "data/interim/unit_features.parquet"), C.Rules()),
                    pd.read_parquet(ROOT / "data/interim/rear_v2.parquet"))
    f = f.merge(pd.read_parquet(ROOT / "data/interim/rear_v3.parquet"), on="unit")
    f["chg_rear_v3"] = (f.rear3_m2 >= RF.REAR_M2) & (f.cover18 >= 0.2)
    f["v3"] = f.chg_new | f.chg_popup | f.chg_rear_v3 | f.chg_demo
    return f


def draw():
    items_path = OUT / "review_v3_fresh_items.csv"
    if items_path.exists():  # the sample is drawn once; never redraw over existing labels
        return pd.read_csv(items_path)
    f = v3_flags()
    seen = set(pd.read_csv(OUT / "review_items.csv").unit)
    pool = f[f.v3 & ~f.unit.isin(seen)]
    s = pool.sample(N, random_state=SEED)[["unit"]].assign(role="precision")
    s["item"] = [f"F{i + 1:02d}" for i in range(len(s))]
    s[["item", "role", "unit"]].to_csv(items_path, index=False)
    print(f"pool {len(pool)} v3-flagged units not yet reviewed; drew {len(s)}")
    return s[["item", "role", "unit"]]


def page(items):
    import review_page as RP

    units = gpd.read_parquet(ROOT / "data/interim/units_19146.parquet")
    rows = items.merge(v3_flags(), on="unit").merge(units[["unit", "address", "opas", "geometry"]], on="unit")
    html = RP.build(rows, "Fresh review, v3 rules — ZIP 19146", "change-review-19146-v3-fresh",
                    "Rules v3 (contiguity + attached-to-building + vehicle mask). None of these units were in the "
                    "first review.", rear_col="chg_rear_v3", export_name="review_v3_fresh_labels.csv")
    (OUT / "review_v3_fresh.html").write_text(html, encoding="utf-8")
    print(f"review_v3_fresh.html: {len(rows)} cases")


def score():
    m = pd.read_csv(OUT / "review_v3_fresh_items.csv").merge(
        pd.read_csv(OUT / "review_v3_fresh_labels.csv").fillna(""), on="item").merge(v3_flags(), on="unit")
    j = m[m.change.isin(["real", "none"])]
    x = (j.change == "real").to_numpy(float)
    by = {}
    for t, col in [("new", "chg_new"), ("popup", "chg_popup"), ("rear", "chg_rear_v3"), ("demolition", "chg_demo")]:
        jj = j[j[col]]
        by[t] = {"n": int(len(jj)), "real": int((jj.change == "real").sum())}
    out = {"n_items": int(len(m)), "n_judged": int(len(j)), "n_unsure": int((m.change == "unsure").sum()),
           "precision": float(x.mean()), "ci95": bootstrap_rate(x), "by_detected_type": by,
           "false_positive_notes": j[j.change == "none"].change_note.tolist()}
    (OUT / "review_v3_fresh_scores.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] == ["score"]:
        score()
    else:
        page(draw())
