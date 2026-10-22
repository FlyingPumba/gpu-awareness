#!/bin/bash
set -euo pipefail
cd /workspace/coding-awareness
if [[ ! -x .venv-coding002/bin/python ]]; then
  /root/.local/bin/uv venv --python /workspace/gpu-fingerprint/.venv/bin/python --system-site-packages .venv-coding002
fi
printf '%s\n' '/workspace/gpu-fingerprint/.venv/lib/python3.11/site-packages' > .venv-coding002/lib/python3.11/site-packages/original-runtime.pth
/root/.local/bin/uv pip install --python .venv-coding002/bin/python -c requirements-overlay.lock -r coding2/requirements-additions.lock
.venv-coding002/bin/python - <<'PY'
import importlib.metadata,json
from pathlib import Path
versions={n:importlib.metadata.version(n) for n in ('torch','transformers','peft','accelerate','safetensors','huggingface-hub','numpy','wandb','pyarrow')}
assert versions['torch']=='2.8.0.dev20250319+cu128'
assert versions['transformers']=='4.57.1'
assert versions['peft']=='0.17.1'
p=Path('runs/coding-002-qwen25-3b');p.mkdir(parents=True,exist_ok=True)
(p/'runtime-setup.json').write_text(json.dumps(versions,indent=2)+'\n')
print(json.dumps(versions))
PY
