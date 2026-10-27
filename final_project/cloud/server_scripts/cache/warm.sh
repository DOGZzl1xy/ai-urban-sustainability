source $HOME/.local/bin/env
export UV_CACHE_DIR=/root/autodl-tmp/cache/uv
cd /root/autodl-tmp/cache
uv venv warm --python 3.13.2 --clear
time VIRTUAL_ENV=/root/autodl-tmp/cache/warm uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
warm/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo WARM_DONE
