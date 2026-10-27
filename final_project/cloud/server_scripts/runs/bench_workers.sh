# Throughput only: 300 frozen steps per worker count, no validation, outputs removed afterwards.
ulimit -n 1048576
cd /root/autodl-tmp/runs/m4/final_project
export POPUP_DATA=citywide POPUP_PREP_CACHE=1 PYTHONIOENCODING=utf-8
Z=$(grep "^Z=" results/citywide/m4/queue_m4.sh | cut -d"\"" -f2)
for W in 6 12 16 24; do
  (while sleep 5; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; done) > /tmp/gpu_w$W.txt &
  SP=$!
  .venv/bin/python -u scripts/unitcls.py train --zips $Z --periods 2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022 \
     --iters 300 --frozen-iters 300 --eval-every 100000 --dead-check 0 --seed 0 --tag bench_w$W \
     --loader fast --workers $W --log-every 50 --save-every 100000 > /tmp/bench_w$W.log 2>&1
  kill $SP
  S=results/citywide/m4/unitcls_bench_w${W}_steps.csv
  echo "workers $W: $(awk -F, "NR>1{print \$1, \$3}" $S | tr "\n" " ")| GPU util mean $(awk "{s+=\$1;n++} END{printf \"%.0f\", s/n}" /tmp/gpu_w$W.txt)%"
  rm -f results/citywide/m4/unitcls_bench_w${W}_* data/interim/models/unitcls_bench_w${W}*.pt
done
echo BENCH_DONE
