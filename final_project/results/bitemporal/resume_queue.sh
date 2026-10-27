#!/usr/bin/env bash
# Resumed v3r / v3b queue (paused 2026-10-22 19:55 for 3 h at the user's request). v3r_s0 is rerun from scratch
# (checkpoints hold weights only, not optimizer state). A run whose unit head is dead at the first evaluation
# (exit 3) is retried with the next seed, at most 3 tries; two live replicates per configuration, interleaved.
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
Z="19132 19121 19143 19125 19130 19123 19122"
ALL="2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022"
run() {  # run NAME PERIODS ITERS FROZEN SEEDS...: first seed whose unit head is alive wins
  local name=$1 periods=$2 iters=$3 frozen=$4; shift 4
  for seed in "$@"; do
    $PY -u scripts/unitcls.py train --zips $Z --periods $periods --iters $iters --frozen-iters $frozen --seed $seed --tag ${name}_s$seed
    rc=$?
    if [ $rc -eq 3 ]; then echo "== ${name}_s$seed dead, next seed"; continue; fi
    [ $rc -ne 0 ] && { echo "== ${name}_s$seed failed rc=$rc"; return 1; }
    $PY -u scripts/unitcls.py dev --tag ${name}_s$seed
    return $?
  done
  echo "== $name: no live run in seeds $*"; return 1
}
# first replicate of each configuration, then the second replicate (different seeds)
run v3r "2018-2022" 6000 3000 0 1 2 && run v3b "$ALL" 8000 4000 0 1 2 && run v3r "2018-2022" 6000 3000 10 11 12 && run v3b "$ALL" 8000 4000 10 11 12
echo "ALL DONE"
