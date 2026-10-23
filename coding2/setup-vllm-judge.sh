#!/bin/bash
set -euo pipefail
cd /workspace/coding-awareness
if [[ ! -x .venv-judge-vllm/bin/python ]]; then
  /root/.local/bin/uv venv --python python3 .venv-judge-vllm
fi
/root/.local/bin/uv pip install --python .venv-judge-vllm/bin/python -r coding2/requirements-vllm.lock
.venv-judge-vllm/bin/python -c 'import vllm,torch,transformers; print(vllm.__version__,torch.__version__,transformers.__version__)'
