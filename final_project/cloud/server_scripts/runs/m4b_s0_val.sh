# When the uploaded m4b_s0 checkpoint arrives: verify sha256, move it into the models folder, re-score validation.
F=/root/autodl-tmp/unitcls_m4b_s0.pt; SHA=e4e2fd19f96b8e9b9f631b94cffa1e3c4dc29036ae14672341e8ffe0d293f51d
P=/root/autodl-tmp/runs/m4/final_project
until [ -f $F ] && [ ! -e $F.tmp ] && [ -s $F ] && [ $(( $(date +%s) - $(stat -c %Y $F) )) -ge 30 ]; do sleep 30; done
if [ "$(sha256sum $F | cut -d" " -f1)" != "$SHA" ]; then mv $F $F.BAD; echo "$(date +%T) BAD sha256: renamed to .BAD, re-upload"; exit 1; fi
mv $F $P/data/interim/models/unitcls_m4b_s0.pt; echo "$(date +%T) OK m4b_s0 checkpoint in place"
ulimit -n 1048576; cd $P; export POPUP_DATA=citywide POPUP_PREP_CACHE=1 PYTHONIOENCODING=utf-8
Z=$(grep "^Z=" results/citywide/m4/queue_m4.sh | cut -d"\"" -f2)
.venv/bin/python -u scripts/unitcls.py val --zips $Z --periods 2018-2022 --tag m4b_s0; echo "$(date +%T) val rc=$? (expect AP pop-up 0.1233 from training)"
