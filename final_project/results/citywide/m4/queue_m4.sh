#!/usr/bin/env bash
# M4 amendment 1 (results/citywide/M4_plan.md): recipe A = M3b schedule (8,000 steps, 4,000 frozen, validation every
# 500), guard at every validation (grew AP < 0.15 -> stop, DEAD_<tag> holds the step, next seed). Two live seeds.
# Fallback: if m4b seeds 0 AND 1 both hit the guard within 2,000 steps -> m4c (positive-pair weight 18.45), two
# seeds, same guard. Nothing else. Resumable: a finished run (training log at step 8000) is skipped; an unfinished
# one continues from its last checkpoint.
# Portable copy of queue_a1.sh (same logic; the Python path is no longer hard-coded to Windows). queue_a1.sh is
# the copy the local run of 2026-10-26 was started from and is left as it is while that run is in progress.
# Cloud: run with POPUP_PREP_CACHE=1 (masks / unit ids read from the package, no citywide rasters needed).
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 POPUP_DATA=citywide
# Python of the project venv: Linux / macOS (.venv/bin) or Windows (.venv/Scripts); override with PY=...
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=.venv/Scripts/python; fi
fi
WORKERS=${WORKERS:-6}
R=results/citywide/m4
Z="19103 19104 19107 19111 19114 19115 19118 19119 19120 19121 19122 19123 19124 19125 19126 19128 19129 19130 19131 19132 19133 19135 19137 19138 19140 19141 19142 19143 19144 19145 19146 19147 19148 19149 19151 19152 19154"
ALL="2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022"
done_run() { [ -f $R/unitcls_$1_train.csv ] && tail -n 1 $R/unitcls_$1_train.csv | grep -q "^8000,"; }
dead_step() { [ -f $R/DEAD_$1 ] && cat $R/DEAD_$1 || echo 0; }
variant() {  # variant NAME POS_WEIGHT -> returns 0 when two live seeds are done
  local name=$1 pw=$2 live=0
  for seed in 0 1 2 3 4 5; do
    [ $live -ge 2 ] && return 0
    tag=${name}_s$seed
    if done_run $tag; then echo "== $tag already done"; live=$((live+1)); continue; fi
    if [ -f $R/DEAD_$tag ]; then echo "== $tag stopped by the guard at step $(dead_step $tag)"; continue; fi
    if [ "$name" = m4b ] && [ $seed -ge 2 ] && [ $(dead_step m4b_s0) -gt 0 ] && [ $(dead_step m4b_s0) -le 2000 ]        && [ $(dead_step m4b_s1) -gt 0 ] && [ $(dead_step m4b_s1) -le 2000 ]; then
      echo "== m4b seeds 0 and 1 both stopped within 2,000 steps: fallback"; return 2
    fi
    echo "== $tag start $(date '+%F %T')"
    $PY -u scripts/unitcls.py train --zips $Z --periods $ALL --iters 8000 --frozen-iters 4000 --eval-every 500         --guard-every-eval --pos-weight $pw --seed $seed --tag $tag --loader fast --workers $WORKERS --log-every 50 --resume
    rc=$?
    echo "== $tag end $(date '+%F %T') rc=$rc"
    if [ $rc -eq 3 ]; then tail -n 1 $R/unitcls_${tag}_train.csv | cut -d, -f1 > $R/DEAD_$tag; continue; fi
    [ $rc -ne 0 ] && return 1
    live=$((live+1))
  done
  [ $live -ge 2 ] && return 0
  echo "== $name: fewer than two live seeds in 0-5"; return 1
}
variant m4b 5
rc=$?
if [ $rc -eq 2 ]; then variant m4c 18.45; rc=$?; fi
echo "QUEUE DONE rc=$rc"
