#!/usr/bin/env bash
# Set up M4 training on a cloud GPU from the package made by cloud/pack.py (see the package README.md).
#
#   bash setup_cloud.sh PKG_DIR [WORK_DIR]      # PKG_DIR holds data/ and code/; WORK_DIR on the data disk
#
# Steps: 1 verify the sha256 of every shard, archive and the weights; 2 unpack code and data; 3 install the locked
# dependencies (uv.lock, Python from .python-version; torch 2.11.0+cu128 from the PyTorch cu128 index, which the
# lock pins for Linux x86_64 as well); 4 print torch / CUDA / GPU; 5 run a 100-step smoke test.
# Training afterwards:  cd WORK_DIR/final_project && POPUP_PREP_CACHE=1 bash results/citywide/m4/queue_m4.sh
set -euo pipefail
PKG=$(cd "${1:?usage: setup_cloud.sh PKG_DIR [WORK_DIR]}" && pwd)
WORK=${2:-/workspace/m4}
mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
need() { command -v "$1" >/dev/null || { echo "missing tool: $1 ($2)"; exit 1; }; }
need sha256sum "coreutils"
need zstd "apt-get install -y zstd"
need tar "tar"
if ! command -v uv >/dev/null; then
  echo "uv not found: installing it (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== 1/5 verify checksums"
(cd "$PKG/data" && sha256sum -c --quiet SHA256SUMS)
(cd "$PKG/code" && sha256sum -c --quiet SHA256SUMS)
echo "all shards, the code archive and the weights match"

echo "== 2/5 unpack into $WORK"
tar -I zstd -xf "$PKG/code/code.tar.zst" -C "$WORK"
P="$WORK/final_project"
for s in "$PKG"/data/shard_*.tar.zst; do
  echo "  $(basename "$s")"
  tar -I zstd -xf "$s" -C "$P"
done
mkdir -p "$P/data/interim/models"
cp "$PKG"/data/dinov3_vitl16_pretrain_sat493m-*.pth "$P/data/interim/models/"
cd "$P"

echo "== 3/5 install the locked environment"
uv python install "$(cat .python-version)"
uv sync --frozen
PY=.venv/bin/python

echo "== 4/5 environment"
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "no CUDA device"
p = torch.cuda.get_device_properties(0)
print("GPU", p.name, "|", round(p.total_memory / 2**30, 1), "GiB")
EOF

echo "== 5/5 smoke test: 100 steps + one validation (dead-check off), outputs removed afterwards"
export POPUP_DATA=citywide POPUP_PREP_CACHE=1 PYTHONIOENCODING=utf-8
Z=$(grep '^Z=' results/citywide/m4/queue_m4.sh | cut -d'"' -f2)
$PY -u scripts/unitcls.py train --zips $Z \
    --periods 2010-2015 2010-2018 2010-2022 2015-2018 2015-2022 2018-2022 \
    --iters 100 --frozen-iters 50 --eval-every 100 --dead-check 0 --seed 0 --tag smoke \
    --loader fast --workers "${WORKERS:-6}" --log-every 10
mkdir -p "$WORK/smoke_test"
mv results/citywide/m4/unitcls_smoke_* "$WORK/smoke_test/" 2>/dev/null || true
rm -f data/interim/models/unitcls_smoke*.pt
echo "smoke test done (logs in $WORK/smoke_test). Train with:"
echo "  cd $P && POPUP_PREP_CACHE=1 nohup bash results/citywide/m4/queue_m4.sh > results/citywide/m4/queue_cloud.log 2>&1 &"
