#!/usr/bin/env bash
# Stage 3 pipeline, run after the 2022 LiDAR of the training ZIPs is on disk. Each step stops the run on failure.
set -eo pipefail
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
for z in 19132 19121 19143 19125; do $PY -u scripts/build_rasters.py 2022 3 $z; done
for z in 19132 19121 19143 19125; do $PY -u scripts/height_dataset.py $z 2022; done
for z in 19132 19121 19143 19125; do $PY -u scripts/lidar_change.py $z 2018 2022; done
$PY -u scripts/train_bitemporal.py anchor
$PY -u scripts/train_bitemporal.py pair
$PY -u scripts/eval_bitemporal.py bitemp_anchor bitemp_pair height_dino_dpt_r025
echo PIPELINE DONE
