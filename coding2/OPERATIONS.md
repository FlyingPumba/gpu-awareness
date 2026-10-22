# coding-002-qwen25-3b: preparation

The user authorized a new experiment on 2026-10-22: Java on the existing RTX
PRO 6000 coordinator (runpod2), C/C++ on the existing A40 worker (runpod1).
After the A40 capacity measurement, the user requested a larger worker GPU
and explicitly authorized `research-pod -a poseidon` for one A100 SXM 80 GB,
at the quoted $1.79/hour Secure Cloud rate plus 100 GB container storage.
The rented Pod is `runpod3`, ID `bmgdovo2b3u6rz`, with the exact requested
GPU, image and 100 GB disk verified. The standard setup command exited 0.
The user subsequently authorized retiring the A40 once it is no longer needed.
Its 622 retained files (623,825,734 bytes) were copied and SHA256-verified under
`outputs/pod-archives/runpod1-a40-20261022/`. The A40 Pod `tis7v9nxk89cws` was
terminated at 2026-10-22T16:23:17Z and verified absent from the account; the A100
and RTX remain running. Do not use the retired `runpod1` alias. The A100 takes
the C/C++ role; W&B uses
the personal account even though the GPU rental uses the Poseidon account.
Use 10,000 distinct problems: 9,800 training and 200 held out. Both GPUs use
the same problem batch, capped by measured replacement-worker capacity, and one shared adapter
updated with equally averaged gradients. Keep the original language-neutral
code-only prompt and fresh pinned Qwen2.5-3B-Instruct base, frozen BF16 base
and FP32 LoRA. Previous experiments and their configurations remain unchanged.

The user selected one **fixed random subset of 10** held-out problems, identical
on both GPUs, for solution-token-loss evaluation and best-checkpoint selection
every 25 updates. After measuring worker capacity, the user approved a maximum
of **1,950 updates at batch 5**, with 100-update linear warmup to 5e-5, cosine
decay to zero, and early stopping after eight non-improving evaluations after
warmup. Select strictly lowest post-update mean per-problem solution-token loss
on the fixed ten problems across both GPUs; exclude baseline from selection.

Generate answers to all 200 held-out problems on the A100 and RTX GPUs before
training and at the selected best checkpoint. Judge these existing rollouts
on the RTX using the latest original pinned Qwen3.8-27B recipe, with options
Python, Java, C/C++, Other. Preserve malformed judgments as unparsed. Do not
execute candidate solutions. Keep raw outputs and provenance.

Metrics and the training trajectory must go to the user's personal W&B account.
The research-pod setup has labeled `WANDB_API_KEY_PERSONAL` and
`WANDB_API_KEY_POSEIDON` credentials but defaults the active key to Poseidon.
The personal key authenticates as `iarcuschin`, confirmed by the user and API.
Telemetry now explicitly selects the personal key for this experiment, leaving
global key defaults unchanged. The two mistaken task-created corporate-key
copies were removed after matching their contents to the known work key.
Personal key files are private and outside the run/artifact directories.
Never print credentials.
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
metrics pending the destination decision. All nine preserved numeric trials
have now been uploaded to personal `iarcuschin/gpu-awareness` and read back:
run `coding-002-qwen25-3b-capacity-worker`, nine matching history rows, zero
logged artifacts. This replay did not rerun any GPU computation. Verification
is saved in `wandb-capacity-verification.json` in the local run directory.
The SDK default API is `api.forge.coreweave.com`; telemetry explicitly uses
`api.wandb.ai`. The earlier project-creation rejection was for `ivan-poseidon`;
creation and logging now succeed for the verified personal `iarcuschin` account.

A100 capacity measurement completed with exit 0 (PID 9342), using the original
eager attention and memory strategy. Its config differs from the A40 capacity
config only in the worker GPU name. Across 12 trials, batches 1 and 2 passed
the longest-sequence and most-solution-token cases; batches 3 and 4 failed on
the longest sequences. Batch 2 peaked at 73.76 GiB. The RTX subsequently passed
eight batch-2 Java stress trials, including coordinator optimizer-state buffers,
with exit 0 and a peak of 78.35 GiB. This verifies a matching batch, not the RTX's
maximum. Both adapters remained unchanged. Local copies have verified hashes;
personal W&B readback matches all 12 A100 and 8 RTX rows, with zero artifacts.
Compact reports are `capacity-a100-summary.json` and `capacity-rtx-summary.json`.

The user questioned the low capacity and did not approve the proposed 4,900
updates. An unchanged-recipe diagnostic (PID 9973, exit 0) measured unique saved
CUDA tensor storage during actual forward/backward, without an optimizer step.
The weights occupy 5.86 GiB. One longest C/C++ example (2,341 tokens) peaks at
39.81 GiB, including 17.64 GiB of saved dense attention matrices across 36 layers.
One median example (711 tokens) peaks at 12.43 GiB; eight distinct median-length
examples (padded to 712 tokens) fit at 58.19 GiB. Thus batch 2 was a worst-case
fixed-batch limit, not a general limit for this model on an A100. The diagnostic
preserves BF16 base, FP32 LoRA, eager attention and the existing loss; the adapter
is unchanged. All 24 original diagnostic files were collected and hash-verified;
all three metric rows were read back from personal W&B, with zero artifacts.
Full evidence is in local `memory-diagnostic-a100/`, with a compact Git report
`memory-a100-summary.json`.

The user approved SDPA for the coding model and explicitly confirmed the
1,792-token cutoff after an earlier suggestion of retaining the 95th percentile.
Use the confirmed cutoff: remove a training pair when either language's full
prompt-plus-reference sequence exceeds 1,792 tokens. This removes 103 of 9,800
training pairs (1.05%), leaving 9,697, and preserves all 200 evaluation records
and the fixed selection subset. Retained training records and their order are
byte-identical; no solution is truncated. Preserve the original frozen data.
Filtered data is `data/coding-002-1792` remotely and local `data-1792/`; the
filter audit and excluded records preserve provenance. Its train SHA256 is
`adb5369c97ee46404943b11f7ec4e241071b3d917baa7f9f459443d5c51b8a67`;
the evaluation SHA256 remains unchanged. `capacity-sdpa-config.json` differs
from the A100 capacity config only in coding-model attention, dataset path and
training data hash. Judge settings, precision and all other recipe fields are
unchanged. Activation checkpointing has not been approved and remains disabled.
The SDPA measurement completed with exit 0 on A100 (PID 10634): batch 5 passed
both stress cases and repeated verification; batch 6 ran out of memory. There
were 18 trials and the highest successful peak was 68.29 GiB. RTX verification
of the matching batch 5 completed with exit 0 (PID 32871), with 10 passing trials
and a 70.92 GiB peak including coordinator optimizer-state buffers. The A100
limits the shared batch; the RTX maximum was not searched.
Both traces show `aten::_scaled_dot_product_efficient_attention` and its backward
operator, rather than the materialized math path. Both fresh adapters remained
unchanged. Configuration, encoded data, base weights, adapter layout and package
versions match across roles. The config canonical SHA256 is
`e4a399bcdf4b67c978dc1114c57584076c12d2d16b23481f5ee5d4a89af30724`.
All 33 collected files per role passed hash verification, and all 18/10 history
rows were read back from personal W&B with zero logged artifacts. Raw results
and frozen sources are in local/remote `capacity-sdpa-a100-worker/` and
`capacity-sdpa-rtx-coordinator/`; corresponding compact summaries are in Git.
The failed dataset transfer was recovered by applying the same committed filter
on both Pods, producing matching hashes. An RTX source check before launch found
only an extra trailing blank line in `model.py`; its original bytes were archived
before matching the canonical file. No GPU job ran during that failed check.

Batch 5 means five distinct shared problems per update, with five C/C++ and five
Java targets. The user approved 1,950 maximum updates (9,750 problem presentations:
all 9,697 retained problems once, then 53 repeats in the next seeded epoch),
retaining the warmup/decay, evaluation and early-stopping settings above.

The approved configuration is `coding2/config.json`, deployment is
`coding2/suite.json`, and the trainer is `coding2/train.py`. Baseline generation
started on both GPUs on 2026-10-22 at 18:34 UTC; training and judging had not
started at that point. Both SDPA measurement processes have exited successfully. The
existing heartbeat is active for this finalized run. The new Pod has a normal `main`
checkout tracking `git@github.com:FlyingPumba/gpu-awareness.git`; pinned runtime
versions and all three batching tests passed before measurement.

The final config canonical SHA256 is
`be3de1a640245a3ba39c50247f76b06c97025642b21c292c646eef1bed1900fe`.
Seven focused CPU checks passed in the real training environment: batch loss
and gradient weighting, seeded ordering/resume, judge schema, trainer batch
integration, the fixed ten selection problems, malformed-label metric handling,
and the approved learning-rate boundaries. Generation, judging and training
log numeric progress/metrics to personal W&B without uploading raw artifacts.
The judge gets its own `.venv-coding002-judge` overlay; its original pinned
numerical packages and original `.venv-judge-qwen38` are preserved.

The baseline sessions are `coding-002-baseline-worker` on runpod3 (initial PID
11368) and `coding-002-baseline-coordinator` on runpod2 (initial PID 33487).
They use `coding/stage-session.sh` and `coding2.evaluate` with the frozen config,
writing `runs/coding-002-qwen25-3b/baseline/`. First actual records were verified
on both GPUs. Personal W&B readback found 11/38 numeric history rows respectively
and zero logged artifacts. The complete matching source/data/model startup
checks are in `results/coding-002-qwen25-3b/baseline-start-verification.json`.
The A100 checkout was verified file-for-file against commit `05fefbd` and is clean.
Its existing SSH identity connects successfully to the RTX coordinator; only
public keys were transferred, with deployment details outside the repository.

Before every next stage, inspect tmux, recorded PID identity, exit/status files,
GPU occupancy and actual JSONL counts. The local scratch monitor is
`work/monitor-coding002.py` relative to the task directory. Do not duplicate or
overlap GPU jobs. Require 200 generations per role and exit 0, collect the worker
files onto the coordinator, and verify candidate identities and hashes. Run
`coding2.judge` on the RTX with inputs `baseline/worker-generations.jsonl` and
`baseline/coordinator-generations.jsonl`, output `baseline/judge`, using the
isolated judge Python and the detached stage wrapper. Preserve judge inputs,
source, config, raw output, and all parse failures. After 400 judgments and exit
0, launch training from the local repository with
`python3 coding2/launch.py coding-002-qwen25-3b`. Monitor actual updates, learning
rates, shared weight hashes, fixed-ten evaluation and checkpoint/optimizer state.
Recovery uses the same run's checkpoint and append-only logs.

After successful training, transfer the selected checkpoint to the A100 with a
matching SHA256 and generate all 200 held-out answers per GPU into `selected/`.
Collect both files and judge 400 candidates into `selected/judge/` on the RTX.
Use `python3 coding2/report.py RUN_DIR OUTPUT_DIR` for the separate C/C++/Java
report, including all four language categories, malformed and truncated counts,
judged correctness and the fixed-ten selection losses. The four-label report's
malformed-exclusion check passed. `coding2/plot_results.py` produces the loss,
language and correctness figures from the final report; inspect the rendered
figures when actual results exist. Use `python -m coding2.publish RUN_ID
--prepare-only` to assemble and review the final bundle, then omit that flag for
authorized new-run HF publication. It retains the selected FP32 adapter, pinned
unmodified base snapshot, datasets, raw candidates/judgments, checkpoints, logs,
source and attribution; internal W&B SDK journals stay local. It checks both
stage completions, all 800 candidate/judgment identities, selection history,
checkpoint/adapter agreement and frozen data hashes, and verifies the uploaded
manifest and file hashes. The loading helper is `coding2/infer.py`; no extra
inference was run. Publication/inference/report imports passed in the actual
environment. A CPU-only synthetic 800-record audit passed and rejected an
altered candidate identity; final publication awaits real completed results.
The old A/B publisher must not be invoked unchanged. Pause the heartbeat only
after all stages, local/W&B/HF verification and Git publication are complete.

The RTX baseline finished at 2026-10-22T18:45:26Z with exit 0: exactly 200
answers, no empty outputs and no token-limit truncations. All prompts, token
decoding and candidate identities passed verification. Personal W&B readback
matched all 200 output-token/progress rows with zero artifacts. Completed files
were collected locally with SHA256 verification; compact evidence is
`results/coding-002-qwen25-3b/baseline-coordinator-verification.json`.
The A100 baseline finished at 2026-10-22T19:05:40Z with exit 0: exactly 200
answers, no empty outputs and one answer at the 1,024-token limit. All prompts,
token decoding and candidate identities passed verification. Personal W&B
readback matched all 200 progress/output-token rows with zero artifacts. All
nine completed role files were collected locally with matching hashes; compact
evidence is `results/coding-002-qwen25-3b/baseline-worker-verification.json`.
The reusable CPU verification helper is local `work/verify-coding002-generation.py`;
run it from the remote root with the coding Python, `PYTHONPATH=.` and
`HF_HOME=/workspace/hf_cache`, passing the stage and role.

Both 200-record baselines and successful exits were verified together on the
coordinator before judging. Baseline judging started on the idle RTX at
2026-10-22T19:26:46Z in tmux `coding-002-baseline-judge`, PID 34576, through
`coding/stage-session.sh`, using `.venv-coding002-judge/bin/python`. The command,
input hashes/identities, source and config are frozen in `baseline/judge/`.
The full Qwen3.8 model is BF16 with eager attention, thinking disabled, greedy
generation and KV cache. Startup verification checked eight actual judgments:
all candidate hashes/identities, blinded messages, prompt tokens, output token
decoding and strict parsing matched. Seven parsed; one reached the judge token
limit and remains explicitly unparsed. Personal W&B readback verified seven
available metric rows against their prefixes and zero logged artifacts. Compact
evidence is `results/coding-002-qwen25-3b/baseline-judge-start-verification.json`.
The A100 is idle; training has not started. Require all 400 judgments and exit 0
before the training launch. Local helper `work/verify-coding002-judge.py` checks
startup or completion without loading model weights; invoke from the remote
root with the judge Python, `PYTHONPATH=.`, `HF_HOME=/workspace/hf_cache`, and
arguments `baseline complete` when the pass finishes. It verifies preservation
of the initial raw prefix and reads back all 400 W&B metric rows on completion.

Commit/push code, configs and compact reports on main after
`scripts/check_git_payload.py`; keep each Git file below 20,000,000 bytes. Keep
raw data/results/checkpoints local, on the existing Pods and in the authorized
new HF artifact repository. Leave the A100 and RTX Pods running.
