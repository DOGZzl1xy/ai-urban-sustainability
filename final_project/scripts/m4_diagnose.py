"""M4 seed 0 diagnosis (2026-10-26): does the model fit the TRAINING units, or only fail on validation?

    POPUP_DATA=citywide python scripts/m4_diagnose.py

Scores two checkpoints of m4_s0 (best grew AP = step 500, and the last checkpoint, step 13,500) on the same number of
random train and val patch pairs of 2018 -> 2022, with the validation scoring of scripts/unitcls.py. Unit AP on train
pairs that is also low points to an optimisation failure; high on train and low on val points to over-fitting or a
train / val difference. Output: results/citywide/m4/diagnose_s0.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bitemporal as B  # noqa: E402
import height_data as H  # noqa: E402
import height_model as M  # noqa: E402
import unitcls as U  # noqa: E402

assert H.CITY, "set POPUP_DATA=citywide"
sys.argv = ["unitcls.py", "dev", "--tag", "m4_s0"]
spec = importlib.util.spec_from_file_location("unitcls_script", ROOT / "scripts/unitcls.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

N = 60
Y0, Y1 = 2018, 2022
sp = pd.read_csv(ROOT / "data/interim/m4_patches.csv", dtype={"zip": str})[["zip", "i", "j", "split"]].drop_duplicates()
zips = sorted(sp[sp.split.isin(["train", "val"])].zip.unique())
pi = B.pair_index(tuple(zips), Y0, Y1).merge(sp, on=["zip", "i", "j"])
pi["uid"] = [str(U.UID_DIR / f"{z}_{i}_{j}.npy") for z, i, j in zip(pi.zip, pi.i, pi.j)]
pi["key"] = [(z, Y0, Y1) for z in pi.zip]
rng = np.random.default_rng(0)
sets = {s: pi[pi.split == s].iloc[np.sort(rng.choice((pi.split == s).sum(), N, replace=False))] for s in ("train", "val")}
need = sorted(set(sets["train"].zip) | set(sets["val"].zip))
sizes, labels = {}, {}
for z in need:
    sizes[(z, Y0, Y1)] = U.unit_id_patches(z, 2018)
    labels[(z, Y0, Y1)] = U.unit_labels(z, Y0, Y1)
out = {}
for name, path, key in [("step500_best_grew", "unitcls_m4_s0_grew.pt", "state"),
                        ("step13500_last", "unitcls_m4_s0_last.pt", "model")]:
    model = U.UnitCls().cuda()
    model.load_state_dict(torch.load(ROOT / "data/interim/models" / path, map_location="cpu", weights_only=False)[key])
    for s, d in sets.items():
        df = S.score_units(model, d, sizes, labels, M.STAGE3_SHIFT)
        r = {"units": len(df), "grew": int(df.grew.sum()), "popup": int(df.popup.sum()),
             "ap_grew": S.ap_score(df.grew, df.logit_grew.to_numpy()),
             "ap_popup": S.ap_score(df.popup, df.logit_popup.to_numpy()),
             "logit_grew_mean": float(df.logit_grew.mean()), "logit_grew_std": float(df.logit_grew.std())}
        out[f"{name}/{s}"] = r
        print(name, s, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    del model
    torch.cuda.empty_cache()
(ROOT / "results/citywide/m4/diagnose_s0.json").write_text(json.dumps(out, indent=2))
