#!/usr/bin/env bash
# Batch 2: verify each archive (uploaded to cloud_pkg_batch2/ or the data-disk root), unpack all, check MANIFEST.csv,
# then run stage-6 inference (results/citywide/stage6_plan.md) over the whole study area.
D=/root/autodl-tmp/pkg_batch2; P=/root/autodl-tmp/runs/m4/final_project
while true; do
  left=0
  while read -r sha name; do
    [ -f "$D/$name" ] && continue
    left=$((left+1))
    for f in /root/autodl-tmp/cloud_pkg_batch2/$name /root/autodl-tmp/$name; do
      [ -f "$f" ] && [ ! -e "$f.tmp" ] && [ -s "$f" ] || continue
      [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 30 ] || continue
      if [ "$(sha256sum "$f" | cut -d" " -f1)" = "$sha" ]; then mv "$f" "$D/$name"; left=$((left-1)); echo "$(date +%T) OK   $name"
      else mv "$f" "$f.BAD"; echo "$(date +%T) BAD  $name (sha256 mismatch, renamed .BAD; re-upload)"; fi
      break
    done
  done < <(tr -d "\r" < $D/SHA256SUMS)
  [ $left -eq 0 ] && break
  sleep 30
done
echo "$(date +%T) all batch-2 archives in; unpacking"
for t in $D/*.tar; do tar -xf "$t" -C $P && echo "  $(basename $t)"; done
M=""; for c in $D/MANIFEST.csv /root/autodl-tmp/cloud_pkg_batch2/MANIFEST.csv /root/autodl-tmp/MANIFEST.csv; do [ -f $c ] && head -1 $c | grep -q archive && grep -q restore.tar $c && M=$c && break; done
if [ -n "$M" ]; then cd $P && tail -n +2 $M | tr -d "\r" | awk -F, "{print \$4\"  \"\$2}" | sha256sum -c --quiet - && echo "$(date +%T) MANIFEST check: all $(($(wc -l < $M)-1)) files match" || { echo "MANIFEST MISMATCH: stopping"; exit 1; }
else echo "$(date +%T) batch-2 MANIFEST.csv not found: archive-level sha256 only"; fi
echo "$(date +%T) BATCH2_READY; stage-6 inference starts"
cd $P && ulimit -n 1048576 && POPUP_DATA=citywide PYTHONIOENCODING=utf-8 .venv/bin/python -u scripts/stage6.py infer --workers 48 2>&1 | grep -v Warning | tee /root/autodl-tmp/runs/stage6_infer.log
