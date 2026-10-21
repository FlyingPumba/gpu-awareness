#!/bin/bash
set -u
cd "$(dirname "$0")/.." || exit 1
role="$1"
output="$3"
mkdir -p "$output"
rm -f "$output/$role.exit"
bash coding/run-role.sh "$@" >> "$output/$role.log" 2>&1 &
trainer_pid=$!
printf '%s\n' "$trainer_pid" > "$output/$role.pid"
wait "$trainer_pid"
result=$?
printf '%s\n' "$result" > "$output/$role.exit"
exit "$result"
