#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
run_id="$1"
export HF_HOME=/workspace/hf_cache
export HF_TOKEN_PATH=/workspace/gpu-awareness-private/hf-token
export PYTHONUNBUFFERED=1
output="runs/$run_id"
rm -f "$output/publish.exit"
/workspace/gpu-fingerprint/.venv/bin/python scripts/publish.py "$run_id" >> "$output/publish.log" 2>&1 &
publisher_pid=$!
printf '%s\n' "$publisher_pid" > "$output/publish.pid"
wait "$publisher_pid"
result=$?
printf '%s\n' "$result" > "$output/publish.exit"
exit "$result"
