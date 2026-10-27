#!/usr/bin/env bash
# M4 amendment 2 (results/citywide/M4_plan.md): config S = recipe A (8,000 steps, 4,000 frozen, validation every 500)
# + sampling weight 18.45, batch 16, EMA 0.999, mean+max unit pooling. Same guard at every validation (grew AP < 0.15
# -> stop, DEAD_<tag> holds the step, next seed), seeds 0-5 one after another until two seeds finish. Resumable like
# queue_m4.sh. Cloud: run with POPUP_PREP_CACHE=1. Afterwards: scripts/m4_compare.py --base m4b_s0 --cand <two tags>.
cd "$(dirname "$0")/../../.."
ulimit -n 1048576 2>/dev/null || true   # DataLoader workers share tensors through file descriptors
export PYTHONIOENCODING=utf-8 POPUP_DATA=citywide
if [ -z "$PY" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=.venv/Scripts/python; fi
fi
WORKERS=${WORKERS:-6}
R=results/citywide/m4
Z="19103 19104 19107 19111 19114 19115 19118 19119 19120 19121 19122 19123 19124 19125 19126 19128 19129 19130 19131 19132 19133 19135 19137 19138 19140 19141 19142 19143 19144 19145 19146 19147 19148 19149 19151 19152 19154"
ALL="2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022"
done_run() { [ -f $R/unitcls_$1_train.csv ] && tail -n 1 $R/unitcls_$1_train.csv | grep -q "^8000,"; }
live=0
for seed in 0 1 2 3 4 5; do
  [ $live -ge 2 ] && break
  tag=m4s_s$seed
  if done_run $tag; then echo "== $tag already done"; live=$((live+1)); continue; fi
  if [ -f $R/DEAD_$tag ]; then echo "== $tag stopped by the guard at step $(cat $R/DEAD_$tag)"; continue; fi
  echo "== $tag start $(date '+%F %T')"
  $PY -u scripts/unitcls.py train --zips $Z --periods $ALL --iters 8000 --frozen-iters 4000 --eval-every 500 \
      --guard-every-eval --pos-weight 18.45 --bs 16 --ema 0.999 --pool meanmax --seed $seed --tag $tag \
      --loader fast --workers $WORKERS --log-every 50 --resume
  rc=$?
  echo "== $tag end $(date '+%F %T') rc=$rc"
  if [ $rc -eq 3 ]; then tail -n 1 $R/unitcls_${tag}_train.csv | cut -d, -f1 > $R/DEAD_$tag; continue; fi
  [ $rc -ne 0 ] && { echo "QUEUE STOPPED rc=$rc"; exit 1; }
  live=$((live+1))
done
if [ $live -ge 2 ]; then echo "QUEUE DONE: two config-S seeds finished"; else echo "QUEUE DONE: config S failed (fewer than two seeds in 0-5)"; fi
