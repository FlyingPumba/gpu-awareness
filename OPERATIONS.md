# Authorized suite operations

The user authorized all five fresh runs in `suite.json`, in order, on the
existing A40/RTX PRO 6000 pair. The 100-step warmup, cosine decay, 5e-5 peak,
1,000-step maximum and eight-evaluation patience rule are approved. Public
HF publication under `iarcuschin` and pushes to `FlyingPumba/gpu-awareness`
are authorized. Preserve the exact configurations and pinned revisions.

The local clone lives in this task's `outputs/gpu-awareness`; local raw copies
belong in the sibling `outputs/suite-runs/<run>/`, outside Git. Remote source
and run artifacts live at `/workspace/gpu-awareness` on both Pods. The shared
Python environment is `/workspace/gpu-fingerprint/.venv/bin/python`. Tokens
and worker-link settings live outside the repository in
`/workspace/gpu-awareness-private/`. Never print or commit their contents.

## Inspect and continue

Use `ssh runpod2` for the coordinator and `ssh runpod1` for the worker. Their
tmux session name is the run ID. Read `runs/<run>/<role>-status.json`, the last
log lines, recorded PID/exit files, `best.json`, `latest_checkpoint.txt` and
`evaluations.jsonl`. Verify that the update number advances, adapter hashes
stay synchronized, learning rates match the schedule, and selected/periodic
checkpoints exist. Log timestamps are UTC. Do not infer a hang from a single
snapshot during model download or loading.

If a run fails, inspect both roles and verify task-owned processes have
exited before restarting. Repair only implementation bugs. Resume this run
from its own latest checkpoint, including optimizer and selection state.
If no checkpoint exists, retain its failed logs and start a fresh replacement
with the identical recipe. Ask before changing experimental semantics.

## Collect and publish each completed run

1. Require `completed` statuses and exit code 0 from both roles. Retrieve
   coordinator artifacts into `outputs/suite-runs/<run>/`. Large raw artifacts
   remain on the Pods and will be uploaded to HF; at minimum retain the
   selected checkpoint locally, plus all JSON/JSONL/logs and evaluation NPZs.
   Retrieve worker JSONL/status/log/exit files from the worker into the same
   local directory and copy those four files to the coordinator run directory.
2. From the local repository, run:

   ```bash
   uv run --offline --no-project --with matplotlib python scripts/report.py ../suite-runs/<run> results/<run>
   ```

   This task's local report dependencies are cached in an isolated uv environment;
   the default Python and training environments do not include Matplotlib.

   Inspect the figure and metrics. Copy `results/<run>/` to
   `/workspace/gpu-awareness/runs/<run>/results/` on the coordinator, so HF
   also receives the report and images. Preserve selected and last metrics
   separately; comparisons to run-001 at a different stopping step are not
   matched-budget comparisons.
3. On the coordinator, start a detached publisher only if its session/process
   is not already active and no verified receipt exists:

   ```bash
   cd /workspace/gpu-awareness
   tmux new-session -d -s '<run>-publish' 'bash scripts/publish-session.sh <run>'
   ```

   Check `publish.log`, `publish.pid`, `publish.exit` and
   `publish-receipt.json`. The publisher exports the best adapter, complete
   pinned base snapshot, all saved checkpoints, logs, provenance, evaluations,
   code and figures. It verifies the selected adapter equals its checkpoint
   and verifies remote file sizes/LFS hashes. Fix actual upload failures and
   retry the same publication; do not silently omit large artifacts.
4. Fetch the verified receipt into `results/<run>/publish-receipt.json`, add
   the live model link/result to README, and commit/push only the requested
   source/result changes. Run `scripts/check_git_payload.py` after staging;
   every Git blob must be below 20,000,000 bytes. Do not commit tokens, model
   weights or unrelated user changes.
5. After the upload is verified, start the next model from fresh initialization:

   ```bash
   python3 scripts/launch.py <next-run-id>
   ```

   Verify actual updates and the first improvement checkpoint. The launcher
   waits for the coordinator model to load before starting the worker. It
   does not replace monitoring. The same Pods are reused sequentially.

## Completion

After all five runs and HF uploads finish, collect final reports and selected
checkpoints, publish the final GitHub comparison and verify the remote commit.
Report selected steps, balanced accuracy, correct-label probabilities and
mean losses; explicitly state that these are the single trained prompt.
Do not overstate a best-selected observation as stable convergence or held-out
generalization. Pause the monitoring heartbeat. Leave both rented Pods
running unless the user explicitly requests otherwise.
