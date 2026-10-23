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
on both GPUs, for evaluation every 25 updates. This includes generated answers
and judging; solution-token loss remains the best-checkpoint selection metric.
The initial implementation omitted interval generation/judging; see the recovery
record below. After measuring worker capacity, the user approved a maximum
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
The A100 remained idle during judging. Require all 400 judgments and exit 0
before the training launch. Local helper `work/verify-coding002-judge.py` checks
startup or completion without loading model weights; invoke from the remote
root with the judge Python, `PYTHONPATH=.`, `HF_HOME=/workspace/hf_cache`, and
arguments `baseline complete` when the pass finishes. It verifies preservation
of the initial raw prefix and reads back all 400 W&B metric rows on completion.

Baseline judging completed with 400 records and exit 0; the verification at
2026-10-22T20:21:37Z checked every candidate identity/hash, blinded message,
prompt token sequence, decoded judgment and strict parse. The initial eight-row
prefix is unchanged. All 400 W&B metric rows matched, with zero artifacts and
the run marked finished. All 19 completed non-W&B-journal files were collected
locally with matching hashes. The compact completion receipt and language
summary are in `results/coding-002-qwen25-3b/`.
The A100 baseline labels are Python 170, Java 0, C/C++ 0, Other 20, unparsed 10;
the RTX labels are Python 168, Java 0, C/C++ 0, Other 20, unparsed 12. There are
34 judge-rated correct answers per role. Of the 22 unparsed judgments, 21
reached the judge token limit. Unparsed records remain outside language and
incorrect counts. These are judge labels, not execution results. Several
inspected Other examples are literal sample outputs instead of implementations.

Fresh training launched through local `coding2/launch.py` on 2026-10-22 at
20:22 UTC, after both GPUs were idle and all 33 frozen source files, the config,
data hashes, 9,697/200 counts and completed judging passed preflight. Both tmux
sessions are `coding-002-qwen25-3b`. The coordinator trainer PID is 35543;
the worker wrapper PID is 12851 and its GPU trainer PID is 12855.
At 20:26:02Z, the first eight actual shared updates passed verification: the
same five problem IDs, C/C++ and Java targets, equal per-problem and role mean
losses, synchronized adapter hashes, and the scheduled learning rates. The
peer provenance matches, with the fresh pinned BF16 base, FP32 adapters and
SDPA. The diagnostic step-zero fixed-ten mean solution-token loss is
0.8508130044 (C/C++ 0.8181457877, Java 0.8834802210); it is excluded from
checkpoint selection. Personal W&B readback verified the available six worker
and seven coordinator update rows, with zero artifacts. Compact evidence is
`training-start-verification.json` in the results directory. Continue monitoring
the actual updates, first step-25 evaluation/checkpoint, selection and both exits.
Local helpers are `work/verify-coding002-training-start.py` and
`work/verify-coding002-first-checkpoint.py`; the latter runs with the training
Python on the coordinator and only reads checkpoint tensors on the CPU.

The first scheduled evaluation and checkpoint at update 25 passed CPU inspection
at 20:27:49Z, while training had advanced through update 32. The same fixed ten
problems yielded mean loss 0.6816656172 (C/C++ 0.6284006983, Java 0.7349305362),
so step 25 became the first eligible selected checkpoint. Its 504 FP32 adapter
tensors match both the synchronized update hash and the standard PEFT export.
All AdamW states are at step 25 with learning rate 1.25e-5; selection and RNG
states are present and verified. Checkpoint SHA256 is
`5b6c3ed6499736136243e756e20bb105e275debb512548eba474a8c2439ac840`.
All 23 checkpoint/export/snapshot files were collected locally and hash-verified.
This is an early fixed-ten loss result; final language outcomes require the
selected-checkpoint generations after training finishes.

Warmup through update 100 passed verification at 20:36:38Z. The first 101
shared updates used the same five problems and agreed per-problem/role loss
averaging. The learning rate reached 5e-5 at update 100 and began cosine decay
at update 101 (4.999996395324314e-5). The update-100 checkpoint has 504 FP32
adapter tensors matching its synchronized weight hash and standard export;
AdamW is at step 100 with the peak learning rate, and selection/RNG state is
preserved. Its SHA256 is
`f3e24abf37d9aa32b38b45e95a659821e400ebf0c7571835e084a054b20450b9`.
At update 100, the fixed-ten mean loss was 0.5197008096 (C/C++ 0.4664163977,
Java 0.5729852214), the lowest through that update, with zero non-improving
evaluations. Personal W&B readback matched all first 100 update rows and five
evaluation rows per role, with zero artifacts. Compact evidence is
`results/coding-002-qwen25-3b/warmup-verification.json`; the CPU-only local
helper is `work/verify-coding002-warmup.py`. Training remains active; read the
live best marker and logs for subsequent selection changes.

Commit/push code, configs and compact reports on main after
`scripts/check_git_payload.py`; keep each Git file below 20,000,000 bytes. Keep
raw data/results/checkpoints local, on the existing Pods and in the authorized
new HF artifact repository. Leave the A100 and RTX Pods running.

On 2026-10-22 the user identified the missing judged evaluations and explicitly
chose to resume and backfill saved checkpoints, retaining loss-based selection.
Training was stopped after update 623. SIGINT was ignored by the background
coordinator process; SIGTERM to its verified PID 35543 stopped it with exit 143,
and the worker exited 1 after peer disconnection. Both GPUs were then verified
idle. Original logs, provenance, exits and source are preserved in
`archive/interval-repair-20261022/{coordinator,worker}` on the corresponding Pod
and locally; all 81/70 archived files passed local SHA256 verification.

Resume this same run from checkpoint 600, SHA256
`15a91fddd7fa5f2c6b627cfbe0fc204afe4a74d16c08987f1fb5e564a84f960e`.
Its optimizer is at update 600, with best step 575, mean loss
0.49485107511281967, and one non-improving evaluation. Updates 601–623 are
replayed from that checkpoint; their original raw records remain preserved.
Twenty historical checkpoints exist: 25, 50, 75, 100, 125, 150, 175, 200, 225,
250, 275, 300, 350, 375, 400, 450, 500, 550, 575 and 600. Judged results for
325, 425, 475 and 525 cannot be reconstructed because no checkpoint was saved.
Keep these gaps explicit; do not interpolate them or restart the experiment.

`coding2/interval_control.py coding-002-qwen25-3b` runs locally with a process
lock and persisted state in the local run's `interval/` directory. First it
resumes 600–625 through `coding2/launch.py --resume ... --pause-after-update 625`,
then verifies the complete step-625 generation/judge path. Next it backfills
the twenty saved checkpoints, and then continues in 25-update segments. Each
boundary saves optimizer/selection state and exits both trainers, generates the
same ten problems on each original GPU, and runs the unchanged pinned judge on
the idle RTX before resuming training. The original 1,950-update LR schedule,
batch semantics, precision, fixed IDs, selection and stopping rules remain
unchanged. The frozen config keeps its original hash for checkpoint recovery;
the additional judged-interval protocol is recorded in `interval/plan.json`,
CLI flags and per-interval frozen source/provenance.

Interval outputs are `interval/step-NNNNNN/`, with ten generations per role,
twenty judgments, successful exit markers and a CPU verification receipt.
`coding2/interval_audit.py` checks candidate/checkpoint identities, prompts,
token decoding, strict parses, artifact hashes and personal W&B readback.
W&B run `coding-002-qwen25-3b-interval-judge-coordinator` uses `update` on the
horizontal axis. Its `interval_eval/worker/*` and `interval_eval/coordinator/*`
metrics contain one completed snapshot per interval: language counts, judged
correctness and unparsed counts. Separate `judge/*` metrics retain candidate
progress. No experiment artifacts are uploaded to W&B. Backfilled points retain their original
training-update coordinates even though they are evaluated later.
All ten focused CPU tests passed before the first repaired launch, including
batching, selection, pause-boundary optimizer/LR preservation and controller
ordering. Verify actual remote progress and first complete interval before
claiming the repair works. Final selected-checkpoint evaluation on all 200
problems per GPU, reporting, local/HF verification and Git publication remain
required after training and interval judging finish.

The first repaired training segment completed at step 625 with both exits 0.
All 25 resumed shared updates, the full-schedule learning rates, fixed-ten
evaluation, synchronized final adapter and AdamW step 625 passed verification.
Checkpoint SHA256 is
`d28cffd30bd2d4462afaf32d7fbfcd766ae7cd98688de5c99cb1aedd1082302d`.
Mean loss at 625 is 0.4949793815612793, so best remains 575 with two
non-improving evaluations. All 25 resumed coordinator metric rows were read
back from personal W&B. Evidence is `interval-resume-verification.json` in the
compact results directory. Initial controller PID 30270 was replaced by PID
32505 while both trainers were paused, to switch checkpoint transfers from the
local upload connection to the existing worker-to-coordinator SSH link. No
optimizer update or generation was repeated during that controller change.
Each checkpoint is still collected locally and hash-verified on both Pods.
Read the current `interval/controller.pid` and process identity before acting.

W&B artifact accounting correction: after a process finishes, the service can
materialize its numeric telemetry as an internal `wandb-history` artifact.
The first such artifact contains only `0000.parquet`; its schema contains the
logged numeric training/evaluation/selection fields and W&B timestamps/steps.
No model, dataset or candidate-output artifact was uploaded. Verification now
reports internal history artifacts separately from user-logged artifacts,
which must remain zero; do not claim a literal zero for the combined API list.

Direct checkpoint transfer succeeded, but controller 32505 then stopped before
generation because its embedded provenance script had an incorrectly escaped
newline. The quoting bug is fixed, stderr is retained for diagnosis, and a
focused regression test compiles the actual generated remote script. That test
passed. Controller 33023 reconnected to the already-paused step 625 and launched
both fixed-ten generation processes with the matching checkpoint; no training
or completed answer was repeated. Current state remains in the controller files.
Report/figure/HF preparation support now includes the judged trajectory and its
explicit missing-checkpoint gaps; final publication requires every recoverable
interval to be present and its verification manifest to match.

The first full interval (625) completed with 10 generations per GPU, 20 judge
records and all exits 0. A100 labels are 8 C/C++ and 2 Java; RTX labels are
7 C/C++, 2 Java and 1 unparsed. Each GPU has one generation at the token limit;
the unparsed RTX judgment reached the judge token limit. Judge-rated correctness
is 2/10 parsed A100 and 2/9 parsed RTX, without executing any solution. These are
the fixed ten examples, not the final all-200 comparison.
Candidate/checkpoint identities, all prompt/output token decoding, blinded
messages, strict parses, full BF16 judge class and frozen judge source passed
verification. The audit explicitly requests list token IDs because Transformers
5.20 otherwise returns a BatchEncoding; all twenty actual generation prompts
matched after this CPU audit compatibility fix, with no regeneration.
Personal W&B readback matched 10 generation rows per role, all 20 judge-progress
rows and one complete `interval_eval/*` snapshot at update 625. All 62 collected
files passed local SHA256 verification. Compact evidence is
`results/coding-002-qwen25-3b/interval-625-verification.json`.
Controller 33023 then began the historical step-25 backfill; training remains
paused at 625 until all twenty saved historical checkpoints are judged. The
report's interval aggregation was checked against these real results and
correctly reports twenty pending intervals plus the four historical gaps.
The final-stage local W&B verification helpers now also distinguish reserved
internal history storage from user-logged experiment artifacts.

The first historical backfill, step 25, completed and passed full verification
at 22:00 UTC: ten generations per GPU, twenty judgments, all exits 0, and 62
local files with matching hashes. A100 labels are 9 Python and 1 unparsed;
RTX labels are 8 Python and 2 unparsed. No generation was truncated; all three
unparsed judgments reached the 512-token limit. These are the same fixed ten
problems used at step 625. Personal W&B readback verified all generation and
judge-progress rows, plus the complete snapshot at update 25. An additional
readback confirmed both update-25 and update-625 snapshots remain present,
despite their evaluation order. There are zero user-logged artifacts; one
reserved internal history artifact was reported for coordinator generation.
Compact evidence is `results/coding-002-qwen25-3b/interval-25-verification.json`.
Controller 33023 advanced to step-50 backfill; training remains paused at 625.

Historical backfill through the end of warmup (steps 25, 50, 75 and 100) is now
complete. Each interval has ten generations per GPU, twenty judgments, successful
exits, full identity/token/parse verification, W&B readback and 62 locally
hash-verified files. The additional compact receipts are
`interval-50-verification.json`, `interval-75-verification.json` and
`interval-100-verification.json` in the results directory.

| Step | GPU | Python | Java | C/C++ | Other | Unparsed | Truncated generations |
|---|---|---:|---:|---:|---:|---:|---:|
| 50 | A100 | 1 | 0 | 9 | 0 | 0 | 4 |
| 50 | RTX | 1 | 0 | 9 | 0 | 0 | 4 |
| 75 | A100 | 0 | 4 | 5 | 0 | 1 | 1 |
| 75 | RTX | 0 | 5 | 4 | 0 | 1 | 2 |
| 100 | A100 | 0 | 3 | 6 | 0 | 1 | 0 |
| 100 | RTX | 0 | 3 | 6 | 0 | 1 | 0 |

These are judge labels on the same fixed ten problems, not final all-200 results.
Every unparsed judgment in these three intervals reached the judge token limit.
All complete snapshots retain their training-update coordinates in personal
W&B; zero user-logged artifacts were found, with internal history storage
reported separately in the receipts. Controller 33023 moved to step-125
backfill. Four of twenty historical checkpoints are complete; training is still
paused at 625 and best remains 575 by the unchanged loss-selection rule.

Historical checkpoints 125, 150, 175 and 200 are also complete, bringing the
backfill to eight of twenty checkpoints at 23:09 UTC. Each has ten generations
per GPU, twenty judgments, successful exits, full identity/token/strict-parse
verification, personal W&B readback and 62 locally hash-verified files. Their
compact receipts are `interval-125-verification.json` through
`interval-200-verification.json` in the results directory.

For all four checkpoints, each GPU's language counts on the fixed ten problems
are 5 C/C++, 4 Java, 0 Python, 0 Other and 1 unparsed. Thus these snapshots do
not show the intended language separation between the GPUs. All eight unparsed
judgments reached the 512-token limit and remain excluded from language and
incorrect counts. Truncated generations for A100/RTX are respectively 1/2 at
125, 1/1 at 150, 3/2 at 175 and 3/2 at 200. Judge-rated correct counts are 0/1,
0/0, 1/2 and 1/1 respectively; no solutions were executed. These ten-problem
snapshots do not replace the final selected-checkpoint evaluation on all 200.
The W&B receipts report zero user-logged artifacts and list reserved internal
history storage separately. Controller 33023 advanced to step-225 generation;
training remains paused at 625 and best remains 575 by solution-token loss.

The historical backfill reached ten of twenty checkpoints at 23:29 UTC, with
steps 225 and 250 fully verified and collected (62 matching local files each).
Each has ten generations per GPU, twenty judgments, all exits 0, matching
candidate/checkpoint/token identities and strict parses, and complete personal
W&B metric readback. Compact receipts are `interval-225-verification.json` and
`interval-250-verification.json`.

| Step | GPU | C/C++ | Java | Unparsed | Truncated generations | Judge-rated correct |
|---|---|---:|---:|---:|---:|---:|
| 225 | A100 | 7 | 3 | 0 | 2 | 1 |
| 225 | RTX | 7 | 2 | 1 | 3 | 2 |
| 250 | A100 | 7 | 2 | 1 | 2 | 1 |
| 250 | RTX | 6 | 4 | 0 | 2 | 1 |

All counts use the same ten problems per GPU; Python and Other counts are zero.
Both unparsed judgments reached the judge token limit. Correctness is judged,
not executed, and unparsed records are not counted as incorrect. W&B has zero
user-logged artifacts, with internal history storage listed separately.
Controller 33023 advanced to step 275; training remains paused at 625 during
backfill, with best step 575 and the original loss-based selection unchanged.

At 2026-10-23T00:18:40Z, fifteen of twenty historical checkpoints were complete.
Steps 275, 300, 350, 375 and 400 each passed generation/judge exit checks,
candidate/checkpoint/token identity and strict-parse checks, personal W&B
readback, and local verification of all 62 collected files. Their compact
`interval-STEP-verification.json` receipts are saved in the results directory.
Step 325 remains an explicit historical gap because its checkpoint was not saved.

| Step | GPU | C/C++ | Java | Unparsed | Truncated generations | Judge-rated correct |
|---|---|---:|---:|---:|---:|---:|
| 275 | A100 | 8 | 1 | 1 | 2 | 1 |
| 275 | RTX | 9 | 1 | 0 | 0 | 1 |
| 300 | A100 | 5 | 2 | 3 | 1 | 0 |
| 300 | RTX | 6 | 2 | 2 | 0 | 1 |
| 350 | A100 | 8 | 1 | 1 | 1 | 2 |
| 350 | RTX | 9 | 1 | 0 | 0 | 2 |
| 375 | A100 | 8 | 2 | 0 | 3 | 2 |
| 375 | RTX | 9 | 1 | 0 | 3 | 2 |
| 400 | A100 | 8 | 2 | 0 | 3 | 1 |
| 400 | RTX | 8 | 2 | 0 | 1 | 2 |

These are the same ten problems per GPU; Python and Other counts are zero.
Six of the seven unparsed judgments reached the 512-token cap. The remaining
step-300 A100 judgment ended after 124 tokens without a closing JSON brace;
the raw output and strict parse failure were inspected and preserved, with no
repair or relabeling. Correctness remains judgment rather than execution.
All generation/progress rows and complete interval snapshots were read back
from personal W&B, with zero user-logged artifacts and internal history storage
reported separately. Step 400 still shows the same language distribution on
both GPUs. Controller 33023 advanced to step 450; training remains paused at
625, with best step 575 selected by the unchanged fixed-ten loss metric.

## Preprocessing fix and user-authorized judging cadence, 2026-10-23

The user requested fixing repeated evaluation setup, then changed future
generation/judging to **every 100 updates**. The same ten held-out problems
remain on each original GPU. After step 625, the next judged checkpoints are
700, 800, 900, and so on. Loss evaluation, strict lowest-loss selection and
eight-evaluation early stopping remain every 25 updates. A terminal training
step between judged intervals proceeds to the existing final selected-best
200-problem-per-GPU evaluation. The numeric config remains unchanged; the
authorization and effective boundary are recorded in `interval/plan.json` and
the compact `interval-cadence-change.json` receipt. Reports and publication now
distinguish the changed cadence from historical missing checkpoints.

`coding2/model.py` now caches deterministic token encodings, keyed by the
configuration, pinned tokenizer representation/template, tokenizer package
versions and encoding source. Original raw data hashes are still checked on
every load. Every cached record, token ID, label, target-token count and order
matched fresh encoding on both Pods and the original training encoding SHA256
`f876ad4887a8bea06ce424cd9af29201e8181b59505d8642d047d69ad4884a1a`.
Warm-cache verification made zero calls to the encoder. Preparation measured
60.63 versus 5.65 seconds on A100 and 35.01 versus 3.28 seconds on RTX.
The cache preserves the original full provenance and numerical recipe and also
avoids retokenization when training resumes. Full checkpoint copies to the local
backup and worker now run concurrently; both copies still finish and pass their
hash checks before generation. This does not claim faster token decoding.

Original source, training provenance and controller state were preserved under
`archive/preprocessing-fix-20261023`. Controller 33023 alone was held during the
deployment; the step-600 judge continued and all three GPU stages exited 0.
After focused CPU cache, cadence, early-stop/reconnect and reporting tests passed,
controller 52258 replaced it and reconnected the existing completed records.
It verified step 600, completing all twenty saved historical backfills, and
started the same run from checkpoint 625 with `--pause-after-update 700`.
The actual trainer startups logged cache hits in 5.55 seconds on A100 and
3.28 seconds on RTX. Both synchronized at 625, reproducing the exact prior mean
loss 0.4949793815612793 before advancing shared five-problem updates. The compact
`cache-resume-verification.json` records matched data/model provenance, actual
learning rates, problem IDs and coordinator weight hashes after the fix.
Read and verify `controller.pid` rather than assuming a PID remains current.

Steps 450, 500, 550, 575 and 600 each have ten generations per GPU and twenty
judgments, successful exits, full token/identity/parse checks, personal W&B
metric readback and 62 locally hash-verified files. Their compact receipts are
saved alongside earlier intervals. The historical gaps remain 325, 425, 475 and
525. Step 575 produced C/C++ on all ten problems on each GPU, with zero
judge-rated correct answers on either GPU; it remains selected by solution-token
loss, not these judgments. No solutions were executed. W&B user-logged artifacts
remain zero; reserved internal history storage is recorded separately.

At 2026-10-23T01:28:19Z, step 700 completed the first judged interval under the
new cadence. Both trainers finished updates 626–700 and exited 0. All 75 shared
five-problem updates and equal role-loss averages were checked; loss selection
ran at 650, 675 and 700, with no generation/judging at 650 or 675. The current
best is step 675, mean solution-token loss 0.4939181588590145; step 700 had one
non-improving evaluation. The step-700 checkpoint and adapter identities agree
on both GPUs, and the controller verified optimizer step 700 and both checkpoint
copies before generation.

Step 700 produced ten generations per GPU and twenty judgments, all stages
exiting 0. The same fixed ten IDs, candidate tokens, strict parses, source and
personal W&B metric readback passed verification, as did all 63 collected local
files. The compact `interval-700-verification.json` receipt includes the training
boundary and real generation cache hits: 5.48 seconds on A100 and 3.10 on RTX.

| GPU | C/C++ | Java | Unparsed | Truncated generations | Judge-rated correct |
|---|---:|---:|---:|---:|---:|
| A100 | 7 | 3 | 0 | 1 | 2 |
| RTX | 6 | 4 | 0 | 2 | 1 |

Python and Other counts are zero. These are interval results on ten problems per
GPU, not the final all-200 evaluation; correctness was judged, not executed.
W&B has zero user-logged artifacts. Its reserved internal `wandb-history`
storage is listed separately in the receipt. Controller 52258 then began the
700-to-800 training segment; future loss checks remain every 25 updates and the
next scheduled generation/judging remains step 800.

At 2026-10-23T01:48:23Z, step 800 passed interval verification and controller
52258 began the 800-to-900 segment. Both trainers had exited 0 at 800; all 100
shared updates 701–800, equal role-loss averages and scheduled learning rates
were verified. Loss evaluations ran at 725, 750, 775 and 800, with no extra
judging at the intermediate steps. Step 800 is the new strict lowest-loss
checkpoint, mean solution-token loss 0.49055520817637444 and zero non-improving
evaluations. Both checkpoint copies and optimizer step 800 were verified before
generation; both GPUs used the same checkpoint and adapter hashes.

The step-800 interval has ten generations per GPU and twenty judgments, all
three stages exiting 0. Each GPU has nine C/C++ classifications and one unparsed
judgment, with no Python, Java or Other classifications. Both unparsed judgments
reached the 512-token cap and remain unparsed. Each GPU has one truncated
generation. Neither GPU has a judge-rated correct answer, compared with two on
A100 and one on RTX at step 700 on the same ten problems. The nine parsed answers
per GPU were judged incorrect; unparsed answers are not counted as incorrect.
No solutions were executed. The loss-based selection rule remains unchanged.

Token/identity/source/strict-parse checks, personal W&B metric readback and all
63 local file hashes passed. W&B user-logged artifacts remain zero, with reserved
internal history storage recorded separately. The compact
`interval-800-verification.json` records these checks and actual generation
cache hits of 5.51 seconds on A100 and 3.37 on RTX. The next scheduled judging
is step 900; loss selection and early stopping still run every 25 updates.

At 2026-10-23T02:08:02Z, step 900 passed interval verification and controller
52258 began the 900-to-1000 segment. Both trainers exited 0 at 900. All 100
shared updates 801–900, equal role-loss averages and scheduled learning rates
were verified. Loss selection ran at 825, 850, 875 and 900 without intermediate
generation/judging. Best remains step 800 at 0.49055520817637444, with four
consecutive non-improving loss evaluations at the step-900 boundary.

Both GPUs have ten generations and the coordinator has twenty judgments; all
three GPU stages exited 0. Each GPU has seven C/C++ classifications, two Java
and one unparsed judgment that reached the 512-token cap. Python and Other are
zero. A100 has one judge-rated correct answer, eight incorrect and three
truncated generations; RTX has zero correct, nine incorrect and two truncated
generations. Unparsed judgments remain separate from incorrect. These are the
same ten-problem interval samples, and no solutions were executed.

Full identity/token/source/strict-parse checks, personal W&B readback and all
63 local file hashes passed. The compact `interval-900-verification.json`
receipt records these checks, zero user-logged W&B artifacts with reserved
internal history storage separately, and generation cache hits of 5.58 seconds
on A100 and 3.14 on RTX. The next scheduled generation/judging is step 1000,
with the unchanged 25-update loss-selection and early-stopping checks between.


On 2026-10-23 the user requested one W&B run for the entire experiment. The
previous job/role suffix created fourteen separate runs and made training and
judge results difficult to compare. The canonical personal run is now
`iarcuschin/gpu-awareness/coding-002-qwen25-3b`, with the display name
`coding-002-qwen25-3b`. Both GPUs and all future stages use shared-mode writers
with distinct labels. Training, loss selection, completed interval judgments,
baseline and final selected results belong to this one run. Coordinator
training metrics keep `train/*`, `eval/*`, `selection/*`; worker training uses
`worker/*`. Completed judgments use `baseline_eval/*`, `interval_eval/*` and
`selected_eval/*`, with `update` as their training-step axis. Detailed generation
and judge progress remain under `generation/{stage}/{role}/*` and
`judge_progress/{stage}/*`. Preparation diagnostics are retained under
`diagnostics/*`. No experiment/model/dataset/raw-output artifact goes to W&B.

All 4,928 historical metric rows through training update 1000 were backed up,
migrated and individually read back, including the baseline and all 24 judged
intervals through 900. Source snapshots, pre-change source and controller hold/
resume records are preserved in `archive/wandb-consolidation-20261023`; migration
plan and receipts are in `wandb-consolidation/`. The source runs are preserved
in the `coding-002-qwen25-3b-legacy` group with a link to the canonical run, and
receive no future writes. Empty parquet columns may read back as omitted sparse
fields; all non-null metric values and row identities are checked. Re-verifying
the migration does not replay records or overwrite newer live summaries.

Only the verified local controller was held while the existing trainers finished
step 1000 with both exits 0. Step 1000 is now selected by the unchanged fixed-ten
loss rule at 0.49021957218647005, with zero non-improving evaluations. Controller
52258 resumed after migration verification and began the scheduled step-1000
generation/judging. No optimizer update or completed generation was repeated.
Read `controller.pid` and actual process/state before acting. The experiment's
100-update judging cadence, 25-update loss selection, precision, optimizer,
data, prompts and final all-200 evaluation remain unchanged.

Per-stage `*-telemetry.json` schema 2 receipts route verification to the canonical
run; historical receipts keep their original provenance. The local final-stage
verification helpers also support both schemas. Stage-level W&B finish flushes
that writer without marking the whole experiment finished. Mark the canonical
run finished only after final evaluation/report/publication are complete.


The single-run transition passed the real step-1000 interval: ten generations
per GPU, twenty judgments, all three exits 0, complete identity/token/parse/
source checks, all 68 local file hashes, and the 10+10 generation rows, twenty
judge-progress rows and one completed snapshot read back from the canonical
run. A100 labels are 8 C/C++ and 2 Java; RTX labels are 7 C/C++ and 3 Java.
Each GPU has one judge-rated correct answer and nine incorrect; neither has
an unparsed judgment. Generation truncations are two on A100 and one on RTX.
These are the same fixed ten problems, with correctness judged, not executed.
The compact receipt is `results/coding-002-qwen25-3b/interval-1000-verification.json`.

At 02:39 UTC the first resumed updates 1001–1003 from both roles were verified
against the actual coordinator loss/learning-rate records in the same W&B run,
while all 4,928 migrated rows remained intact. The experiment group contains
exactly one run; fourteen preserved source runs are in its legacy group.
Five focused telemetry/migration tests and eight interval/controller tests pass.
The browser view was checked under the personal account and the interval,
validation-loss and training-loss chart axes were set to `update`. The repair
changes logging and presentation only. Controller 52258 is running the
1000-to-1100 segment; the next scheduled judged interval is 1100, unless the
unchanged loss-based early-stopping rule ends training first. The heartbeat
remains active for the remaining final evaluation and publication work.


At 2026-10-23T02:57:26Z, controller 52258 resumed training from 1100 toward
1200 after the step-1100 interval passed its full audit. Both preceding
trainers exited 0. All 100 shared updates 1001–1100, equal per-problem and
role-loss averages, scheduled learning rates, optimizer step 1100 and the
checkpoint adapter hash were verified. Fixed-ten loss evaluation ran at 1025,
1050, 1075 and 1100; best is step 1050 at 0.48948836252093314, with two
non-improving evaluations at the 1100 boundary. Step-1100 mean loss is
0.4906560517847538. There was no intermediate generation/judging.

The scheduled interval produced ten answers per GPU and twenty judgments, all
three stages exiting 0. A100 has nine C/C++ labels, one Java, one judge-rated
correct and nine incorrect; RTX has seven C/C++, two Java, one unparsed,
zero correct and nine incorrect. Python and Other are zero on both. RTX's
unparsed judgment reached 512 tokens and remains separate from incorrect.
Both GPUs have two truncated generations. Compared with step 1000 on the same
ten problems, judged correctness stayed 1/10 on A100 and declined from 1/10 to
0/10 on RTX. Correctness was judged, not executed; checkpoint selection remains
based only on fixed-ten solution-token loss.

All token/identity/source/strict-parse checks and 69 local file hashes passed.
Actual generation cache hits took 5.61 seconds on A100 and 3.21 on RTX. At
03:00 UTC, all 10+10 generation rows, twenty judge-progress rows, one completed
interval snapshot and resumed updates 1101–1112 from both roles were read back
from the single canonical W&B run. All 4,928 migrated rows still match their
preserved originals, the main group contains one run, and all fourteen legacy
runs remain preserved. User-logged artifacts and internal history artifacts
both currently number zero. The compact receipt is
`results/coding-002-qwen25-3b/interval-1100-verification.json`. The next scheduled
judged interval is 1200, subject to the unchanged 25-update early-stopping
checks. Final selected-best all-200 evaluation and publication remain pending.
