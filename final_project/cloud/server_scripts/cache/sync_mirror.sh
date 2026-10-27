set -e
source $HOME/.local/bin/env
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
C=/root/autodl-tmp/cache
cd /root/autodl-tmp/runs/m4/final_project
uv export --frozen --no-emit-project --no-header --no-annotate -o $C/req_all.txt >/dev/null
# lock sources: torch + torchvision from the PyTorch cu128 index, everything else from PyPI (here: Aliyun mirror of PyPI)
awk "/^(torch|torchvision)==/{skip=1; print > \"$C/req_torch.txt\"; next} /^[a-z0-9]/{skip=0} skip{print > \"$C/req_torch.txt\"; next} {print > \"$C/req_pypi.txt\"}" $C/req_all.txt
echo "pypi: $(grep -c == $C/req_pypi.txt)  torch: $(grep -c == $C/req_torch.txt)"
uv venv .venv --python 3.13.2 --allow-existing
export VIRTUAL_ENV=$PWD/.venv
time uv pip install --require-hashes --no-deps -r $C/req_pypi.txt --index-url https://mirrors.aliyun.com/pypi/simple
time uv pip install --require-hashes --no-deps -r $C/req_torch.txt --index-url https://download.pytorch.org/whl/cu128
unset VIRTUAL_ENV
time uv sync --frozen
.venv/bin/python -c "import torch, torchvision, timm, numpy, pandas; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.get_device_name(0), numpy.__version__, pandas.__version__)"
echo SYNC_DONE
