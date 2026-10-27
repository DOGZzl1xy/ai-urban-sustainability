#!/usr/bin/env bash
# Remaining runs with the fast data path: v3b seed 0, then v3r and v3b seed 1 (fallback seeds if a head is dead).
# Safe to re-run after an interruption: a finished run (dev json present) is skipped, an unfinished one resumes
# from its last periodic checkpoint (--resume). A dead unit head (exit 3) moves to the next seed (max 3 seeds).
cd "$(dirname "$0")/../.."
export PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python
WORKERS=${WORKERS:-6}
Z="19132 19121 19143 19125 19130 19123 19122"
ALL="2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022"
run() {  # run NAME PERIODS ITERS FROZEN SEEDS...
  local name=$1 periods=$2 iters=$3 frozen=$4; shift 4
  for seed in "$@"; do
    tag=${name}_s$seed
    if [ -f results/bitemporal/unitcls_${tag}_dev.json ]; then echo "== $tag already done"; return 0; fi
    if [ -f results/bitemporal/DEAD_$tag ]; then echo "== $tag dead earlier, next seed"; continue; fi
    echo "== $tag start $(date '+%F %T')"
    $PY -u scripts/unitcls.py train --zips $Z --periods $periods --iters $iters --frozen-iters $frozen \
        --seed $seed --tag $tag --loader fast --workers $WORKERS --log-every 50 --resume
    rc=$?
    echo "== $tag train end $(date '+%F %T') rc=$rc"
    if [ $rc -eq 3 ]; then touch results/bitemporal/DEAD_$tag; continue; fi
    [ $rc -ne 0 ] && return 1
    $PY -u scripts/unitcls.py dev --tag $tag || return 1
    echo "== $tag dev end $(date '+%F %T')"
    return 0
  done
  echo "== $name: no live run in seeds $*"; return 1
}
run v3b "$ALL" 8000 4000 0 2 4 && \
run v3r "2018-2022" 6000 3000 1 3 5 && \
run v3b "$ALL" 8000 4000 1 3 5
echo "QUEUE DONE $(date '+%F %T')"
