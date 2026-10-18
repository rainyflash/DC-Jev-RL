#!/bin/bash
# Rebuilds the experiment environment on a fresh Colab GPU runtime (or any Linux box with a recent NVIDIA GPU).
#   bash scripts/setup_colab.sh            # everything below, idempotent
# - kev's own uv-locked environment (torch 2.8) with the kernels of kev's Modal image, MuJoCo/gymnasium and math-verify;
# - a separate vLLM 0.31 environment for rollouts (vLLM pins its own torch);
# - the pinned model snapshots and the math datasets.
# W defaults to /content/work; if Google Drive is mounted, runs/ is linked to Drive so results survive a runtime reset.
set -euxo pipefail
W=${W:-/content/work}
HERE=$(cd "$(dirname "$0")" && pwd)
KEV_COMMIT=${KEV_COMMIT:-1f04cfa28e6a708485b28a9585010b9e436b776e}
mkdir -p $W/logs $W/hf $W/code $W/tmp
# Exact package versions of the experiments (uv pip freeze of both environments on session 1, scripts/env/). Without
# them uv resolves the newest compatible releases: a rebuild a day later already got a newer filelock.
grep -v '^-e \| @ ' $HERE/env/kev_freeze.txt > $W/tmp/kev_constraints.txt
export HF_HOME=$W/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TRITON_CACHE_DIR=$W/hf/triton-cache UV_LINK_MODE=copy
command -v uv || pip install -q uv
if [ -d /content/drive/MyDrive ] && [ ! -e $W/runs ]; then
  mkdir -p /content/drive/MyDrive/dcjev/runs && ln -s /content/drive/MyDrive/dcjev/runs $W/runs
fi
mkdir -p $W/runs
cd $W
[ -d kev ] || git clone -q https://github.com/jaredpalmer/kev.git
cd kev && git checkout -q $KEV_COMMIT && echo "KEV_COMMIT=$(git rev-parse HEAD)"
uv sync --extra serve --python 3.13
PY=$W/kev/.venv/bin/python
uv pip install --python $PY -c $W/tmp/kev_constraints.txt "flash-linear-attention==0.5.2" "triton>=3.7.1"
uv pip install --python $PY --no-deps "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
uv pip install --python $PY -c $W/tmp/kev_constraints.txt "mujoco==3.16.0" "gymnasium[mujoco]==1.3.0" "matplotlib>=3.9" "pandas>=2.2" "scipy>=1.14" "pytest" "math-verify" "datasets>=3"
[ -x $W/vllm-env/bin/python ] || uv venv $W/vllm-env --python 3.13
uv pip install --python $W/vllm-env/bin/python -c $HERE/env/vllm_freeze.txt "vllm==0.31.0" "math-verify" ninja
$PY - <<'PYEOF'
import torch, transformers, peft, fla, triton, mujoco, gymnasium
print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())
print("transformers", transformers.__version__, "peft", peft.__version__, "fla", fla.__version__, "triton", triton.__version__)
print("mujoco", mujoco.__version__, "gymnasium", gymnasium.__version__)
import causal_conv1d, math_verify; print("causal_conv1d, math_verify OK")
from huggingface_hub import snapshot_download
for repo, rev, pats in [("Qwen/Qwen3.5-4B-Base", "1001bb4d826a52d1f399e183466143f4da7b741b", None),
                        ("jaredpalmer/kev-4b", "6cfce5c2fa4b4bd64026336ab649c5ca78857d52", ["*.json", "*.safetensors", "*.pt", "*.txt", "*.jinja"]),
                        ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", None)]:
    print("downloaded", repo, rev, snapshot_download(repo, revision=rev, allow_patterns=pats))
PYEOF
$W/vllm-env/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
if [ -d $W/code/DC-Jev-RL/dcjev ] && [ ! -f $W/data/math500.jsonl ]; then
  (cd $W/code/DC-Jev-RL && PYTHONPATH=. $PY -m dcjev.llm.data --out $W/data)
fi
uv pip freeze --python $PY > $W/runs/env_kev_freeze.txt; uv pip freeze --python $W/vllm-env/bin/python > $W/runs/env_vllm_freeze.txt
echo SETUP_DONE
