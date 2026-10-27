"""Score review_labels.csv (exported from review.html) and write the final verdict.

Precision: real / (real + none) on the 50 precision samples ("unsure" reported separately).
Unpermitted confirmed: candidates marked both "real change" and "confirmed no permit".
Verdict (FEASIBILITY_TEST.md §0): recall new >= 70% and upward >= 70% (any change detected, primary set) and
confirmed unpermitted >= 10 -> continue; recall ok but < 10 confirmed -> reframe as "where densification happens";
recall below bar -> stop and discuss.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from match import bootstrap_rate  # noqa: E402

OUT = ROOT / "results/popup_feasibility"
items = pd.read_csv(OUT / "review_items.csv")
lab = pd.read_csv(OUT / "review_labels.csv", dtype=str).fillna("")
m = items.merge(lab, on="item", how="left").fillna("")
p = m[m.role == "precision"]
judged = p[p.change.isin(["real", "none"])]
x = (judged.change == "real").to_numpy(dtype=float)
lo, hi = bootstrap_rate(x)
u = m[m.role == "unpermitted"]
confirmed = int(((u.change == "real") & (u.permit == "no_permit")).sum())
rec = pd.read_csv(OUT / "recall_by_type.csv")
r = rec[(rec.set == "primary") & (rec.measure == "detected_any")].set_index("kind").recall
recall_ok = bool(r["new"] >= 0.7 and r["upward"] >= 0.7)
verdict = ("continue" if recall_ok and confirmed >= 10 else
           "reframe: where densification happens" if recall_ok else "stop and discuss")
res = {
    "precision": float(x.mean()) if len(x) else None, "precision_ci": [lo, hi], "precision_n_judged": int(len(x)),
    "precision_unsure": int((p.change == "unsure").sum()), "precision_unlabelled": int((p.change == "").sum()),
    "false_positive_reasons": p[p.change == "none"].change_note.value_counts().to_dict(),
    "unpermitted_reviewed": int((u.permit != "").sum()), "unpermitted_confirmed": confirmed,
    "unpermitted_found_permit": int((u.permit == "found").sum()),
    "unpermitted_no_real_change": int((u.change == "none").sum()),
    "match_failure_reasons": u[u.permit == "found"].permit_note.value_counts().to_dict(),
    "recall_new": float(r["new"]), "recall_upward": float(r["upward"]), "verdict": verdict,
}
(OUT / "review_scores.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
