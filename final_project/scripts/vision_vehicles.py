"""Vehicle detection on the exported aerial chips (runs inside the popup-vision Docker image, GPU).

Model: Ultralytics YOLO11-OBB pretrained on DOTA v1 (aerial imagery; classes include "small vehicle" and
"large vehicle"). Writes data/interim/chips/vehicles.json: {chip: [{"cls", "conf", "poly": [[col, row] x 4]}]}.

docker run --rm --gpus all -v <final_project>:/work popup-vision python scripts/vision_vehicles.py
"""

import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
CHIPS = ROOT / "data/interim/chips"
WEIGHTS = ROOT / "data/interim/models/yolo11m-obb.pt"
WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
VEHICLE = {"small vehicle", "large vehicle"}
CONF = 0.25

model = YOLO(str(WEIGHTS) if WEIGHTS.exists() else "yolo11m-obb.pt")
if not WEIGHTS.exists() and Path("yolo11m-obb.pt").exists():
    Path("yolo11m-obb.pt").replace(WEIGHTS)
names = model.names
print("classes:", names)
files = sorted(CHIPS.glob("*_20*.png"))
out = {}
for i in range(0, len(files), 32):
    batch = files[i:i + 32]
    for f, r in zip(batch, model.predict([str(p) for p in batch], imgsz=1024, conf=CONF, verbose=False, device=0)):
        dets = []
        if r.obb is not None and len(r.obb):
            for cls, conf, poly in zip(r.obb.cls.tolist(), r.obb.conf.tolist(), r.obb.xyxyxyxy.tolist()):
                if names[int(cls)] in VEHICLE:
                    dets.append({"cls": names[int(cls)], "conf": round(conf, 3), "poly": [[round(x, 1), round(y, 1)] for x, y in poly]})
        out[f.stem] = dets
    print(f"{min(i + 32, len(files))}/{len(files)}", flush=True)
(CHIPS / "vehicles.json").write_text(json.dumps(out))
n = sum(len(v) for v in out.values())
print(f"{len(out)} chips, {n} vehicles ({sum(d['cls'] == 'large vehicle' for v in out.values() for d in v)} large)")
