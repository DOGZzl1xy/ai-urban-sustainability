ulimit -n 1048576
cd /root/autodl-tmp/runs/m4/final_project
export POPUP_DATA=citywide POPUP_PREP_CACHE=1 PYTHONIOENCODING=utf-8
Z=$(grep "^Z=" results/citywide/m4/queue_m4.sh | cut -d"\"" -f2)
(while sleep 3; do nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits; done) > /tmp/smoke_s_gpu.txt &
SP=$!
.venv/bin/python -u scripts/unitcls.py train --zips $Z --periods 2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022 \
  --iters 100 --frozen-iters 50 --eval-every 100 --dead-check 0 --seed 0 --tag smoke_s --loader fast --workers 6 \
  --log-every 10 --bs 16 --pos-weight 18.45 --pool meanmax --ema 0.999
echo "train rc=$?"
kill $SP
echo "peak GPU MiB $(sort -t, -k1 -n /tmp/smoke_s_gpu.txt | tail -1)  mean util $(awk -F, "{s+=\$2;n++} END{printf \"%.0f\", s/n}" /tmp/smoke_s_gpu.txt)%"
.venv/bin/python -u scripts/unitcls.py val --zips $Z --periods 2018-2022 --tag smoke_s; echo "val rc=$?"
cp data/interim/unitcls_preds/smoke_s_valbest_2018_2022.parquet data/interim/unitcls_preds/smoke_t_valbest_2018_2022.parquet
.venv/bin/python scripts/m4_compare.py --base smoke_s --cand smoke_t smoke_s --n 200 | grep -E "diff|ci95|adopt|final|blocks|val_" ; echo "compare rc=$?"
rm -f results/citywide/m4/unitcls_smoke_s_* results/citywide/m4/m4_compare.json data/interim/models/unitcls_smoke_s*.pt data/interim/unitcls_preds/smoke_*
echo SMOKE_S_DONE
