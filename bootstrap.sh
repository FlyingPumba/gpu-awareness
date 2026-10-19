#!/bin/bash
set -euo pipefail
# Run inside the experiment directory on the matching RunPod image.
# Torch and its CUDA dependencies come from the image; never re-resolve them.
/root/.local/bin/uv venv --python "$(command -v python)" --system-site-packages .venv
/root/.local/bin/uv pip install --python .venv/bin/python --no-deps -r requirements-overlay.lock
.venv/bin/python -c 'import torch,transformers,peft; assert torch.__version__ == "2.8.0.dev20250319+cu128"; print(torch.__version__,transformers.__version__,peft.__version__)'
