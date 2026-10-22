# coding-002-qwen25-3b: preparation

The user authorized a new experiment on 2026-10-22: Java on the existing RTX
PRO 6000 coordinator (runpod2), C/C++ on the existing A40 worker (runpod1).
Use 10,000 distinct problems: 9,800 training and 200 held out. Both GPUs use
the same problem batch, capped by measured A40 capacity, and one shared adapter
updated with equally averaged gradients. Keep the original language-neutral
code-only prompt and fresh pinned Qwen2.5-3B-Instruct base, frozen BF16 base
and FP32 LoRA. Previous experiments and their configurations remain unchanged.

The user selected one **fixed random subset of 10** held-out problems, identical
on both GPUs, for solution-token-loss evaluation and best-checkpoint selection
every 25 updates. The user chose to decide the update budget **after measuring
A40 batch capacity**. Do not launch training or freeze the schedule before that
decision. The previous learning rate, optimizer, warmup/decay and stopping
recipe remain the proposal baseline, not a newly approved update budget.

Generate answers to all 200 held-out problems on both original GPUs before
training and at the selected best checkpoint. Judge these existing rollouts
on the RTX using the latest original pinned Qwen3.8-27B recipe, with options
Python, Java, C/C++, Other. Preserve malformed judgments as unparsed. Do not
execute candidate solutions. Keep raw outputs and provenance.

Metrics and the training trajectory must go to the user's personal W&B account.
The available local credential authenticates as `ivan-poseidon`, but project
creation in that namespace fails with `Entity ivan-poseidon not found`. The
read API returns a placeholder entity with ID -1, so its lookup was not evidence
of write access. The account lists `poseidon-research` as its team. A question
is pending about the correct personal destination/credential or permission to
use that team. Do not silently switch destinations. Never print credentials.
W&B receives
metrics/config/system telemetry, not raw dataset or model artifacts. The user
authorized HF artifact publication for this new experiment. Do not reinterpret
that as authorization to publish the older coding-001 raw artifacts.

Both Pods were idle before preparation. The coordinator has a new isolated
`/workspace/coding-awareness/.venv-coding002` overlay for wandb and pyarrow,
with original training package versions verified unchanged. The original
training and Qwen3.8 judging environments are preserved.

Pinned CodeContests collection finished successfully: 13,328 source problems,
10,113 Java/C++ pairs, 39 hash-verified shards (7,509,753,086 bytes). The initial
text audit leaves 8,240 pairs after normalized-statement deduplication, no
explicit language hints, and the previous 600-word limit. A supplemental pinned
LeetCode source contains 2,778 Java/C++ pairs with descriptions. Data preparation
must preserve source attribution, deduplicate across the combined pool, measure
lengths and inspect the final count before creating the split or profiling it.
The combined pool yielded 10,986 distinct eligible candidates. Selecting the
10,000 with shortest maximum reference-token length produced 7,249 CodeContests
and 2,751 LeetCode problems. Frozen files are in remote `data/coding-002` on both
Pods and locally in `outputs/coding-runs/coding-002-qwen25-3b/data/`.
The compact `results/coding-002-qwen25-3b/data-summary.json` records exact hashes,
selection IDs, attribution, filters and lengths. Longest sequences are 2,341
tokens for C/C++ and 2,439 for Java. No reference was truncated or executed.

The A40 capacity measurement completed nine trials with the original eager
attention recipe: batch 1 passed both longest-sequence and most-solution-token
stress cases, each repeated three more times; batch 2 ran out of memory on the
two longest sequences. Batch 1 peaked at 42,757,341,696 allocated bytes
(39.82 GiB). The fresh adapter hash stayed unchanged. Measurements and the
exact used source are backed up locally under `capacity-worker/`.
After writing all measurements and its summary, the process exited 1 because
the final status log duplicated the `role` argument. This logging bug is fixed
in the current source; retain the original exit/logs and do not describe that
process as exit 0. The numerical trials and saved summary were completed.

The first capacity launch failed during W&B initialization before model loading;
its logs are preserved remotely in `archive/capacity-worker-wandb-init/`.
The completed capacity measurement explicitly used W&B offline mode to preserve
metrics pending the destination decision. Nothing has been synced to the cloud.
The SDK default API is `api.forge.coreweave.com`; telemetry explicitly uses
`api.wandb.ai`, where personal project creation was also rejected.

A second question is pending: measure capacity with activation checkpointing
to permit larger physical batches, or retain the current strategy and batch 1.
Do not enable activation checkpointing before the user's choice. Then return
to the requested update-budget decision using measured batch capacity and
9,800 training problems. The main training configuration and schedule are not
frozen, and no training, baseline generation or judging has started.
The existing heartbeat remains paused; activate it for the finalized new run
when its stages are ready. Both Pods remain running and idle after profiling.

Commit/push code, configs and compact reports on main after
`scripts/check_git_payload.py`; keep each Git file below 20,000,000 bytes. Keep
raw data/results/checkpoints local, on the existing Pods and in the authorized
new HF artifact repository. Leave both rented Pods running.
