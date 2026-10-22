#!/bin/bash
set -euo pipefail
cd /workspace/coding-awareness
if [[ ! -x .venv-coding002-judge/bin/python ]]; then
  /root/.local/bin/uv venv --python .venv-judge-qwen38/bin/python --system-site-packages .venv-coding002-judge
fi
printf '%s\n' '/workspace/coding-awareness/.venv-judge-qwen38/lib/python3.11/site-packages' '/workspace/gpu-fingerprint/.venv/lib/python3.11/site-packages' > .venv-coding002-judge/lib/python3.11/site-packages/original-judge-runtime.pth
/root/.local/bin/uv pip install --python .venv-coding002-judge/bin/python -c coding2/requirements-judge.lock -c coding2/requirements-additions.lock wandb==0.31.0
.venv-coding002-judge/bin/python - <<'PY'
import importlib.metadata
import json
from pathlib import Path
import subprocess
names = ['torch', 'transformers', 'numpy', 'safetensors', 'huggingface-hub', 'accelerate']
versions = {n: importlib.metadata.version(n) for n in names}
original = json.loads(subprocess.check_output([
    '.venv-judge-qwen38/bin/python', '-c',
    'import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in ' + repr(names) + '}))'], text=True))
assert versions == original, (versions, original)
for line in Path('coding2/requirements-judge.lock').read_text().splitlines():
    name, expected = line.split('==')
    assert versions[name] == expected
versions['wandb'] = importlib.metadata.version('wandb')
directory = Path('runs/coding-002-qwen25-3b')
directory.mkdir(parents=True, exist_ok=True)
(directory / 'judge-runtime-setup.json').write_text(json.dumps(versions, indent=2) + '\n')
print(json.dumps(versions))
PY
