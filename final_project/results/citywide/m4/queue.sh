#!/usr/bin/env bash
# M4 (results/citywide/M4_plan.md): seed 0 then seed 1, 41,000 steps each, validation every 2,500 steps and at
# step 500 (dead-head check: grew AP < 0.15 -> DEAD_<tag>, next unused seed). Resumable: rerunning continues an
# unfinished run from its last checkpoint (--resume); a finished run (training log at step 41000) is skipped.
cd "$(dirname "$0")/../../.."
export PYTHONIOENCODING=utf-8 POPUP_DATA=citywide
PY=.venv/Scripts/python
R=results/citywide/m4
Z="19103 19104 19107 19111 19114 19115 19118 19119 19120 19121 19122 19123 19124 19125 19126 19128 19129 19130 19131 19132 19133 19135 19137 19138 19140 19141 19142 19143 19144 19145 19146 19147 19148 19149 19151 19152 19154"
ALL="2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022"
done_run() { [ -f $R/unitcls_$1_train.csv ] && tail -n 1 $R/unitcls_$1_train.csv | grep -q "^41000,"; }
live=0
for seed in 0 1 2 3 4 5; do
  [ $live -ge 2 ] && break
  tag=m4_s$seed
  if done_run $tag; then echo "== $tag already done"; live=$((live+1)); continue; fi
  if [ -f $R/DEAD_$tag ]; then echo "== $tag dead earlier"; continue; fi
  echo "== $tag start $(date '+%F %T')"
  $PY -u scripts/unitcls.py train --zips $Z --periods $ALL --iters 41000 --frozen-iters 20500       --eval-every 2500 --dead-check-at 500 --seed $seed --tag $tag --loader fast --workers 6 --log-every 50 --resume
  rc=$?
  echo "== $tag end $(date '+%F %T') rc=$rc"
  if [ $rc -eq 3 ]; then touch $R/DEAD_$tag; continue; fi
  [ $rc -ne 0 ] && exit 1
  live=$((live+1))
done
echo "QUEUE DONE"
