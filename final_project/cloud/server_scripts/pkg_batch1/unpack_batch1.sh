#!/usr/bin/env bash
# After collect_batch1.sh has all 14 archives: unpack them into the project and check every file against MANIFEST.csv.
D=/root/autodl-tmp/pkg_batch1; P=/root/autodl-tmp/runs/m4/final_project
until grep -q "ALL BATCH-1" $D/collect_batch1.log; do sleep 60; done
echo "$(date +%T) all archives in; unpacking"
for t in $D/*.tar; do tar -xf "$t" -C $P && echo "  $(basename $t)"; done
M=$D/MANIFEST.csv
[ -f $M ] || for c in /root/autodl-tmp/cloud_pkg_batch1/MANIFEST.csv /root/autodl-tmp/MANIFEST.csv; do [ -f $c ] && cp $c $M; done
if [ -f $M ]; then
  cd $P && tail -n +2 $M | tr -d "\r" | awk -F, "{print \$4\"  \"\$2}" | sha256sum -c --quiet - && echo "$(date +%T) MANIFEST check: all $(($(wc -l < $M)-1)) files match"
else
  echo "$(date +%T) MANIFEST.csv not uploaded: archive-level sha256 only (already checked)"
fi
du -sh $P/data/interim/anchor_rasters $P/data/interim/anchor_units $P/data/raw/ortho/tiles 2>/dev/null
echo "$(date +%T) BATCH1_READY"
