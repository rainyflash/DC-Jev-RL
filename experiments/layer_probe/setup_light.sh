#!/bin/bash
# The part of scripts/setup_colab.sh that probe.py needs: kev's environment with the Qwen3.5 kernels (no vLLM, no
# MuJoCo), the base policy at its pinned revision, and the math datasets (for the training problems' text).
#   bash setup_light.sh   (on a Colab session, after push_code.sh)
set -euxo pipefail
W=/content/work
mkdir -p $W/logs $W/hf $W/tmp $W/data
export HF_HOME=$W/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TRITON_CACHE_DIR=$W/hf/triton-cache UV_LINK_MODE=copy HF_HUB_OFFLINE=0
command -v uv || pip install -q uv
grep -v '^-e \| @ ' $W/code/DC-Jev-RL/scripts/env/kev_freeze.txt > $W/tmp/kev_constraints.txt
cd $W
[ -d kev ] || git clone -q https://github.com/jaredpalmer/kev.git
cd kev && git checkout -q 1f04cfa28e6a708485b28a9585010b9e436b776e
uv sync --extra serve --python 3.13
PY=$W/kev/.venv/bin/python
uv pip install --python $PY -c $W/tmp/kev_constraints.txt "flash-linear-attention==0.5.2" "triton>=3.7.1" "datasets>=3"
uv pip install --python $PY --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
$PY -c "import torch, transformers, peft, fla, causal_conv1d; print('torch', torch.__version__, 'transformers', transformers.__version__, 'fla', fla.__version__, torch.cuda.get_device_name(0))"
$PY -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-4B-Base', revision='1001bb4d826a52d1f399e183466143f4da7b741b'))"
cd $W/code/DC-Jev-RL && PYTHONPATH=. $PY -m dcjev.llm.data --out $W/data
echo SETUP_DONE
