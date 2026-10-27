#!/usr/bin/env bash
# Wait until collect_shards.sh has all 16 shards in pkg/data, then run setup_cloud.sh (verify, unpack, env, smoke test).
source $HOME/.local/bin/env
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
until grep -q "ALL 16 SHARDS" /root/autodl-tmp/pkg/collect_shards.log; do sleep 60; done
echo "$(date +%T) all shards in; running setup_cloud.sh"
time bash /root/autodl-tmp/pkg/code/setup_cloud.sh /root/autodl-tmp/pkg /root/autodl-tmp/runs/m4
echo "$(date +%T) SETUP_EXIT=$?"
