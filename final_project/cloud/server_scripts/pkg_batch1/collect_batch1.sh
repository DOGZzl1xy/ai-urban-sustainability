#!/usr/bin/env bash
# Move finished batch-1 uploads from /root/autodl-tmp into pkg_batch1/ after a sha256 check. Exits when all are in.
SRC=/root/autodl-tmp/cloud_pkg_batch1; DST=/root/autodl-tmp/pkg_batch1
while true; do
  left=0
  while read -r sha name; do
    [ -f "$DST/$name" ] && continue
    left=$((left+1))
    f="$SRC/$name"
    [ -f "$f" ] && [ ! -e "$f.tmp" ] && [ -s "$f" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 30 ] || continue
    if [ "$(sha256sum "$f" | cut -d" " -f1)" = "$sha" ]; then
      mv "$f" "$DST/$name"; left=$((left-1)); echo "$(date +%T) OK   $name"
    else
      mv "$f" "$f.BAD"; echo "$(date +%T) BAD  $name (sha256 mismatch, renamed to .BAD; re-upload it)"
    fi
  done < <(tr -d "\r" < "$DST/SHA256SUMS")
  [ $left -eq 0 ] && { echo "$(date +%T) ALL BATCH-1 ARCHIVES IN $DST"; break; }
  sleep 30
done
