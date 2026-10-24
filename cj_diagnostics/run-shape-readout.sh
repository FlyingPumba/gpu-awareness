#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
role="$1"
shift
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1
python_bin=/workspace/coding-awareness/.venv-coding002/bin/python
if [[ "$role" == coordinator ]]; then
    exec "$python_bin" -m cj_diagnostics.shape_readout --role "$role" "$@"
fi
source /workspace/gpu-awareness-private/worker-link.env
ssh -N -L 127.0.0.1:29872:127.0.0.1:29872 \
    -i "$PAIR_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$PAIR_KNOWN_HOSTS" \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p "$PAIR_SSH_PORT" "$PAIR_SSH_TARGET" &
tunnel_pid=$!
trap 'kill "$tunnel_pid" 2>/dev/null || true' EXIT
sleep 2
kill -0 "$tunnel_pid"
"$python_bin" -m cj_diagnostics.shape_readout --role "$role" "$@"
