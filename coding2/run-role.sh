#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
role="$1"
config="$2"
output="$3"
shift 3
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/workspace/gpu-awareness-private/hf-token}"
export PYTHONUNBUFFERED=1
python_bin="${GPU_AWARENESS_PYTHON:-/workspace/coding-awareness/.venv-coding002/bin/python}"
if [[ "$role" == coordinator ]]; then
    exec "$python_bin" -m coding2.train --role "$role" --config "$config" --output "$output" "$@"
fi
# Deployment-specific SSH details live outside the public repository.
source /workspace/gpu-awareness-private/worker-link.env
ssh -N -L 127.0.0.1:29871:127.0.0.1:29871 \
    -i "$PAIR_SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$PAIR_KNOWN_HOSTS" \
    -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -p "$PAIR_SSH_PORT" "$PAIR_SSH_TARGET" &
tunnel_pid=$!
trap 'kill "$tunnel_pid" 2>/dev/null || true' EXIT
sleep 2
kill -0 "$tunnel_pid"
"$python_bin" -m coding2.train --role "$role" --config "$config" --output "$output" "$@"
