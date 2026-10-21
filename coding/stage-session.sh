#!/bin/bash
# Run a generation/judging stage in tmux with persistent logs and exit status.
set -u
cd "$(dirname "$0")/.." || exit 1
output="$1"
name="$2"
shift 2
mkdir -p "$output"
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-/workspace/gpu-awareness-private/hf-token}"
export PYTHONUNBUFFERED=1
"$@" >> "$output/$name.log" 2>&1 &
stage_pid=$!
printf '%s\n' "$stage_pid" > "$output/$name.pid"
wait "$stage_pid"
result=$?
printf '%s\n' "$result" > "$output/$name.exit"
exit "$result"
