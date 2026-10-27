#!/usr/bin/env bash
# Move finished shard uploads from /root/autodl-tmp into pkg/data/ after a sha256 check. Exits when all are in.
SRC=/root/autodl-tmp; DST=/root/autodl-tmp/pkg/data
while true; do
  left=0
  while IFS=, read -r sha name bytes; do
    [ "$name" = shard ] && continue
    [ -f "$DST/$name" ] && continue
    left=$((left+1))
    f="$SRC/$name"
    [ -f "$f" ] && [ ! -e "$f.tmp" ] || continue
    [ "$(stat -c %s "$f")" = "$bytes" ] || continue
    [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge 30 ] || continue
    if [ "$(sha256sum "$f" | cut -d" " -f1)" = "$sha" ]; then
      mv "$f" "$DST/$name"; left=$((left-1)); echo "$(date +%T) OK   $name"
    else
      mv "$f" "$f.BAD"; echo "$(date +%T) BAD  $name (sha256 mismatch, renamed to .BAD; re-upload it)"
    fi
  done < <(tr -d '' < "$DST/SHARDS.csv")
  [ $left -eq 0 ] && { echo "$(date +%T) ALL 16 SHARDS IN $DST"; break; }
  sleep 30
done
