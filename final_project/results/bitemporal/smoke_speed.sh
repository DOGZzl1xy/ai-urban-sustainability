#!/usr/bin/env bash
# 200-step speed / loss comparison of the old data path and the new one (same config, seed 0), run after v3r_s0.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
Z="19132 19121 19143 19125 19130 19123 19122"
for L in old fast; do
  nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -l 1 > results/bitemporal/smoke_gpu_$L.txt &
  SMI=$!
  START=$(date +%s)
  $PY -u scripts/unitcls.py train --zips $Z --periods 2018-2022 --iters 200 --frozen-iters 3000 --eval-every 100000 \
      --seed 0 --loader $L --workers 6 --log-every 10 --tag smoke_$L
  echo "loader $L: wall $(( $(date +%s) - START )) s (includes setup and the final validation)"
  kill $SMI
done
echo SMOKE DONE
