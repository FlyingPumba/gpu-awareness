# Paired C/Python experiment

## Current state

Training completed all 1,000 updates on both Pods at 2026-10-21T23:32:24Z,
with both role exits 0. The selected checkpoint is step 850: C validation loss
0.5763642670, Python 0.6092171283, mean 0.5927906977. The terminal step-1000
mean was 0.5928224958; training reached the maximum with six non-improving
evaluations, rather than stopping early. The final learning rate is zero.
All 1,000 paired updates, the native learning-rate schedule, all 41 validation
evaluations and strict selection replay passed verification. The selected and
terminal checkpoint optimizer/selection states and exact FP32 adapter exports
were checked. All 150 collected training artifact hashes matched locally;
worker-native provenance is retained separately from the coordinator's earlier
worker-provenance copy in `verification/training-completion-worker/`.
See `training-completion-verification.json` and the compact
`coding/results/coding-001-qwen25-3b/training-complete.json`.

The selected checkpoint and export are backed up locally. The exact checkpoint
was copied to the A40; both Pods and the local copy match SHA-256
`b0e718c4c8135bfb95e390fdc576256d0f6e5c201b537fbae8e4a70649415346`.
The synchronized FP32 adapter hash is
`8bdfb1518232c850c8252558b4439b9c8c21442da8233f0c88d135952648938e`.
Selected generation started at 2026-10-22T09:15:24Z on A40 (PID 23411) and
09:15:28Z on RTX (PID 28872), in `coding-001-selected-worker` and
`coding-001-selected-coordinator`. Actual generated records and identities
were verified on both GPUs. Use `runs/coding-001-qwen25-3b/selected/` and
checkpoint `checkpoint-000850.pt`; both roles use KV caching throughout.
An initial launch exited before creating stage files because the Pod shell
changed to `/workspace` despite tmux's `-c`; the corrected tmux command starts
with `cd /workspace/coding-awareness && exec bash coding/stage-session.sh ...`.
The two launch receipts and original attempts are retained in `verification/`.
No model outputs were produced by those first attempts.
Do not restart completed baseline work or training. Wait for 200 selected
answers and exit 0 per role, collect and verify both files, then run the fixed
Qwen3-8B judge on all 400 selected answers. Final reporting and verification
remain pending; keep the heartbeat active and both rented Pods running.
The reporting code now includes per-role generation truncation, empty outputs,
actual cache counts and judge token-limit counts, while keeping malformed
judgments separate. Use `coding/plot_results.py` with the final `report/results.json`
to render training loss, language and correctness figures. The training-loss
figure is already rendered and inspected; final comparison figures await judging.

The code-only baseline is complete: the A40 finished 200/200 at 16:44:03Z
and the RTX finished 200/200 at 16:23:26Z, both with exit 0. All 400 answers
ended with EOS; none reached the output limit without EOS. All fourteen role
artifacts were collected locally and their SHA-256 hashes verified; the worker
files were also copied to the coordinator and verified there. See
`baseline-completion-verification.json`. Worker answers 1–136 and all 200
coordinator answers used no cache; worker answers 137–200 used KV caching.
The exact 136-answer worker prefix remains unchanged.
The baseline Qwen3-8B judge was relaunched on the coordinator at 16:49:10Z in
tmux session `coding-001-baseline-judge` after fixing a generation-default bug.
The first 84 judgments were sampled because Transformers 4.57.1 replaced the
GenerationConfig default-valued do_sample=False with Qwen3's saved default.
They are preserved under `archive/baseline-judge-sampling-20261021/` and must
not enter the current results. `judge.py` now passes do_sample=False directly
to generate and records it in provenance. The installed config-resolution path
confirmed GREEDY_SEARCH after this fix, and confirmed the Qwen2.5 baseline was
already greedy (its older saved Transformers version uses legacy defaults).
See `judge-greedy-fix-verification.json`. A warning about model defaults may
still appear before explicit keyword overrides are applied; that warning does
not describe the final resolved sampling setting. Completed outputs are under
`runs/coding-001-qwen25-3b/baseline/judge/`.
The repaired judge completed all 400 judgments with exit 0. All parsed, and
none reached the 512-token limit. A40 language counts were C 1 / Python 188 /
Other 11; RTX counts were C 1 / Python 189 / Other 10. The judge rated 73/200
answers correct on each GPU (36.5%). Raw outputs and six judge artifacts were
collected locally with matching SHA-256 hashes; see
`baseline-judge-completion-verification.json`. Compact summaries are in
`coding/results/coding-001-qwen25-3b/`. Do not restart generation or judging.

Fresh training has now been launched with `python3 coding/launch.py
coding-001-qwen25-3b`, using the approved config without --resume. Both Pods
passed idle-process, exact source/config and dataset hash checks. The exact
source and launch recipe are in `training-launch-manifest.json`. Monitor
tmux session `coding-001-qwen25-3b` on each Pod and the run-directory root
status, logs, shared update hashes, learning rates and checkpoints.
Actual paired updates through step 3 were verified: same problem on both
GPUs, equal role averaging, FP32 adapters, and learning rate
1.5e-06 at the latest verified update. Initial validation losses were
C 1.05785322, Python 1.45103747, mean 1.25444535.
See `training-start-verification.json`. The first best checkpoint at step 25
has now been verified and backed up locally, including its FP32 adapter and
optimizer state. Validation losses fell to C 0.89423830, Python 1.16788839,
mean 1.03106335. The saved optimizer is at step 25 with learning rate 1.25e-05;
selection records best step 25 and zero non-improving evaluations. The standard
adapter export exactly matches the checkpoint and synchronized update hash.
All 30 backup file hashes matched, and paired progress through step 38 was
verified. See `first-checkpoint-verification.json` and
`first-checkpoint-backup-verification.json`; the compact summary is
`coding/results/coding-001-qwen25-3b/first-checkpoint.json`.
Training is still running; continue monitoring subsequent evaluations and exits.

Warmup completed at step 100 with the planned learning rate 5e-05. Validation
losses were C 0.61425592, Python 0.66530987, mean 0.63978290, selecting step 100
with zero non-improving evaluations. Steps 50, 75 and 100 passed checkpoint,
optimizer, selection and exact FP32 adapter-export checks. The step-100 checkpoint
and adapter are backed up locally; all 37 warmup artifact and log hashes matched.
Paired updates through step 101 verified the start of cosine decay. See
`verification/checkpoint-000100.json`, `warmup-backup-verification.json` and
`coding/results/coding-001-qwen25-3b/warmup.json`. Continue the same training run;
selected-checkpoint generation and judging remain pending until training ends.

At 17:53:49Z the worker exited 1 after its SSH tunnel timed out and the gradient
connection reset. The coordinator remained stalled at update 126; its verified
process was stopped with SIGTERM (exit 143). Both roles' logs, status, exits and
source/config were preserved under `archive/network-interruption-20261021T175349Z/`.
After confirming both GPUs idle, the same run resumed from its verified step-125
checkpoint with unchanged settings. Recovery is verified through paired step 128:
the restored validation losses matched exactly on both GPUs, and replayed step
126 produced the original weight hash. Updates returned to about 21–22 seconds.
Current PIDs are worker wrapper 22119 / trainer 22123 and coordinator 24705;
the tmux session remains `coding-001-qwen25-3b` on both Pods. Use the existing
`scripts/run_history.py` recovery-aware history for progress and reporting;
raw logs retain the rolled-back step 126 and the diagnostic resume evaluation.
See `recovery-20261021T175349Z.json` locally and the compact committed
`coding/results/coding-001-qwen25-3b/recovery-20261021.json`.

Step 200 completed at 18:27:33Z, with validation and checkpoint saving finished
at 18:27:59Z. Validation losses were C 0.58954741, Python 0.63154435 and mean
0.61054588, selecting step 200 with zero non-improving evaluations. The saved
optimizer is at step 200 with learning rate 4.849231551964771e-05; its FP32
adapter export exactly matches the checkpoint and synchronized update hash.
The full checkpoint, adapter, validation records and both roles' log snapshots
are backed up locally with verified hashes. See
`step-000200-backup-verification.json`, `verification/milestone-000200/` and
the compact `coding/results/coding-001-qwen25-3b/step-000200.json`. Training
continues under the same recipe; generation and judging of the selected
checkpoint remain pending until training ends.

The best checkpoint moved to step 275 at 18:56:32Z, with validation losses
C 0.58660201, Python 0.62719270 and mean 0.60689736. Steps 300 and 325 had
slightly higher means (0.60705643 and 0.60740539), leaving the non-improvement
count at two as of step 325. Both trainers continued normally through paired
step 335. Checkpoints 275 and 300 passed FP32 adapter, optimizer and selection
verification. The scheduled step-300 checkpoint and adapter are backed up
locally; all 53 snapshot/artifact hashes matched. See
`step-000300-backup-verification.json`, `verification/milestone-000300/` and
`coding/results/coding-001-qwen25-3b/step-000300.json`. Local backup audits use
`verification/native-learning-rate-reference.json` for exact schedule comparison;
the Mac CPU math library differs by one bit at step 303, while both trainers'
rates match exactly. Training code and its numerical settings are unchanged.

Step 400 selected a new best checkpoint at 19:44:15Z, with validation losses
C 0.58254439, Python 0.62021501 and mean 0.60137970. The non-improvement count
reset to zero. Its FP32 adapter, synchronized weight hash, optimizer step and
learning rate (3.7500000000000003e-05), and selection state passed verification.
The complete checkpoint, adapter and both roles' log snapshots are backed up
locally; all 61 file hashes matched and paired updates through step 409 were
verified. See `step-000400-backup-verification.json`,
`verification/milestone-000400/` and the compact
`coding/results/coding-001-qwen25-3b/step-000400.json`. Both trainers continue
under the same recipe. Final generation and judging remain pending.

Step 500 selected a new best checkpoint at 20:22:22Z: C validation loss
0.58106116, Python 0.61635417, mean 0.59870766, with zero non-improving
evaluations. The FP32 adapter export and synchronized update hash matched;
optimizer step 500, learning rate 2.9341204441673266e-05 and selection state
passed verification. All 69 local backup file hashes matched, including the
complete checkpoint, adapter and both roles' log snapshots. Paired updates
through step 509 and all 200-problem validation records through step 500 were
verified. See `step-000500-backup-verification.json`,
`verification/milestone-000500/` and
`coding/results/coding-001-qwen25-3b/step-000500.json`. Training continues;
selected-checkpoint generation and judging have not started.

Step 600 selected a new best checkpoint at 21:00:30Z: C validation loss
0.58125295, Python 0.61250568, mean 0.59687931, with zero non-improving
evaluations. Its FP32 adapter export, synchronized weight hash, optimizer step
600, learning rate 2.0658795558326743e-05 and selection state passed verification.
The full checkpoint, adapter and both roles' log snapshots are backed up locally;
all 77 file hashes matched. Paired updates through step 606 and the complete
validation history through step 600 were verified. See
`step-000600-backup-verification.json`, `verification/milestone-000600/` and
`coding/results/coding-001-qwen25-3b/step-000600.json`. Both trainers continue
under the agreed recipe; final generation and judging remain pending.

At approximately 16:22Z the user requested vLLM for inference generation.
The user then reaffirmed that FP32 is important because the experiment hinges
on small numerical differences, asking whether vLLM could preserve it. They
did not authorize a BF16 inference copy. vLLM 0.32.0's standard GPU LoRA kernels
accept BF16/FP16 adapter weights, so there is no supported switch preserving
the current BF16-base/FP32-LoRA path. The user subsequently explicitly requested KV caching on 2026-10-21.
Only generation.use_cache changes from false to true; engine, model and adapter
precision, batching, prompt, data and training remain unchanged. No numerical
equivalence study is requested. Continue this authorized recipe; vLLM remains
a proposal requiring a compatible FP32 adapter implementation.
The current code-only coordinator baseline completed 200/200 at 16:23:26Z
with exit 0, all answers ending in EOS. Its process ended and GPU memory was
released. All seven artifacts are backed up locally with matching remote
SHA-256 hashes in `baseline-coordinator-completion-verification.json`.
The worker resumed unfinished baseline generation with KV caching. Retain all
completed baseline answers and their original per-record cache/config identities,
including the completed uncached coordinator baseline. Pre-change worker files,
source and config are preserved in archive/baseline-before-kv-cache-20261021/.
The worker resume explicitly used --resume-uncached-config with that archived
config; only the false-to-true cache setting may differ. Do not regenerate the
completed answers. Selected-checkpoint generation will use KV caching throughout.

The original five-model suite is complete and its heartbeat is paused.
The user requested code-only output on 2026-10-21 at approximately 16:02Z.
Only the shared prompt in `common.py` and `config.json` changed; it still does
not name a programming language. Baseline generation restarted on both GPUs at
approximately 16:05Z so baseline, training and selected-checkpoint evaluation use this prompt.
The completed generation sessions were
`coding-001-baseline-worker` on runpod1 and `coding-001-baseline-coordinator`
on runpod2. Their outputs are under
`/workspace/coding-awareness/runs/coding-001-qwen25-3b/baseline/`.
The initial FP32 adapter hash matches on both GPUs:
`a7c9e3dc42330bff473577954753921d3bbae099fdb271dc956b2182b48dbdf9`.
Both frozen base hashes match the original Qwen2.5-3B run:
`7fdd4eeb373e7b8c86baa9cfd230a22a8f237780a25820476b0c3a00bb1e5525`.
The new five-minute heartbeat `monitor-paired-c-and-python-coding-experiment`
is active. The old suite heartbeat remains paused.
The superseded baseline started on both GPUs at 2026-10-21T12:08Z. Its worker saved its first solution (850 tokens)
at 12:11:51Z; coordinator had saved five solutions by the first local backup.
Both GPUs' encoded dataset, base, adapter and runtime provenance match. Initial
generation/provenance snapshots were backed up locally.
The coordinator completed all 200 baseline generations at 2026-10-21T14:03:33Z
and exited 0. Its process/session ended and GPU memory was released. All seven
coordinator artifacts were copied locally and matched remote SHA-256 hashes;
195 answers ended with EOS and five reached the 1,024-token limit. Verification
is saved in the archived `baseline-coordinator-completion-verification.json`.
The worker was deliberately stopped at 135 answers for the requested prompt
change and its wrapper recorded exit 143; the coordinator retained exit 0.
Both old stage directories, logs, exact prompt source and hashes are preserved
under `archive/baseline-explanations-20261021T160233Z/` in each remote run and the
local run directory. These old answers must not enter the new baseline judge.
The root `manual-baseline-inspection.json` describes this superseded prompt.
Wait for 200 answers and exit 0 from each new baseline role before judging.
The user has approved:

- Fresh Qwen/Qwen2.5-3B-Instruct, revision
  `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Code-only output without explanations or Markdown fences, requested after
  inspecting the original baseline; keep the language unspecified.
- 1,000 total problems, split into 800 training / 200 held-out.
- Keep 1,000 short problems; some intermediate tasks are okay.
- The previous 1,000-update maximum, 100-step warmup to 5e-5 then cosine
  decay, evaluation every 25 updates, strict lower-loss selection, and
  stopping after eight non-improving evaluations after warmup.
- Qwen3-8B judging only; no generated-solution execution against tests.
- Generation and judging before training and at the selected best checkpoint.
  Checkpoint selection uses mean solution-token loss on all held-out problems.

The difficulty decision is resolved: the user chose 1,000 short problems with
some intermediate tasks. The finalized score-at-most-500 subset is ranked by
the longer C/Python reference. Scores are a source-specific proxy, not a
guarantee of beginner difficulty; unscored AIZU problems were also eligible.
The median longer reference is 150 tokens, maximum 418; maximum prompt plus
reference was 1,965 tokens with the original prompt and is 1,983 with the new prompt.
All 2,000 reference token sequences and EOS targets were verified unchanged;
training and generation prefixes match. The split is unchanged. A C++ word-boundary filter bug was fixed before
freezing the split, excluding p02406 and p03978 from the candidate pool.

Final data and audit: `outputs/coding-data/coding-001/` in this task.
Train SHA-256: `f3b78ab0d92bb3b2de211c10b9194717e99473048d818354decf6855e4565985`.
Eval SHA-256: `45988107bcb1d35b6907b37659a85c7ebf05d44de5948d6546f768ab886ef2a5`.
All 2,000 reference hashes and Accepted-language metadata were checked.
Problem IDs, known duplicate groups and exact reference code are disjoint
between train and validation. The final descriptions have no detected explicit
language hints; representative descriptions, samples and references were inspected.

## Source and data collected

New source is in this `coding/` directory. Parent `train.py`,
`training_policy.py`, old configs, results and suite operations are unchanged.
The coding optimizer/transport/recovery loop preserves the previous loop;
its changes are the data/loss/evaluation calls and matching the problem ID
across the paired gradients. Both models still share the exact same adapter.
A40 receives C targets; RTX PRO 6000 receives Python targets for the same
problem at each update. Training keeps one example per GPU and no accumulation.
At 800 training problems, 1,000 paired updates cover at most 1.25 passes.

The official CodeNet 1.0.0 archive is on the coordinator:

`/workspace/coding-awareness/source-data/Project_CodeNet.tar.gz`

Source URL:
https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz

Archive SHA-256:
`7e96620307f874269a62cc0225f2a5a13d2de2b3c368cf61319c87ec276620fb`

Official documentation: https://github.com/IBM/Project_CodeNet
Inspected documentation revision: `55c323b527ab3d6510d55edff188bc0a0d7bc5e5`.

`source-data/metadata-extract/Project_CodeNet/` contains descriptions and
submission metadata. `source-data/reference-extract/Project_CodeNet/` contains
3,412 accepted C/Python 3 references and the supplied identical-problem groups.
`source-data/candidates.json` contains 1,706 English, text-only paired candidates.
All 1,706 pairs passed Python-reference syntax and tokenization checks after
fixing chat-boundary token merging. No solution code was executed.

Collection currently chooses the submission at the 25th percentile of accepted
source sizes for each problem/language, ordered by size then submission ID.
The exploratory collector excludes missing English descriptions, image-based
statements and explicit programming-language/template hints. `prepare_data.py` preserves original code and
submission hashes, collapse duplicate groups and create a seeded problem-level
80/20 split using the agreed short-problem selection.

The score-300 exploratory audit is at:
`source-data/validated-candidates/data-audit.json`.
It found 1,006 unique text candidates after the duplicate/length checks, but
includes advanced unscored problems; it is not a final dataset.

## Training and judging implementation

`common.py` creates the language-neutral user request:

```text
Please implement the following coding problem. Output only the implementation code, nothing else. Do not include explanations or Markdown code fences.

{description}
```

The native Qwen chat prefix is tokenized identically to generation. Completion
tokens are encoded separately to prevent a leading code newline from merging
with the chat-prefix newline. Loss masks the entire prompt and supervises the
reference code plus assistant EOS. No target truncation is implemented.

The final config is `config.json`, with verified dataset hashes. Its canonical
SHA-256 is `444e10e91d7bb021f463bfda26c35f2be3facdfa8bb774ce86db4e3be02c0899`.
The pre-cache code-only config SHA-256 was
`73043a5d824c2cbfff4c1f625d33de31f2364a6564bf7aa89b10443c369cf392`;
completed baseline records retain that identity.
Compared with the previous Qwen2.5-3B config, existing fields change only for
the prompt, target labels, solution-token loss, held-out selection metric and
the subsequently requested generation KV cache.
Data paths resolve relative to `/workspace/coding-awareness`.

`evaluate.py` generates one greedy answer per held-out problem on each original
GPU, before training and after loading the selected checkpoint. Generation
uses a 1,024-new-token cap, batch size one and
KV caching, using the original BF16/FP32/eager path. Completed uncached baseline
answers are retained with their actual settings recorded. Raw token IDs, text,
completion status, config/adapter/checkpoint hashes and timings are retained.

`judge.py` uses the original Qwen/Qwen3-8B revision
`b968826d9c46dd6066d109eabc6255188de91218` on the RTX PRO 6000, BF16/eager,
with thinking disabled. The judge sees problem description and candidate text,
without GPU identity, reference solution, desired language or training step.
It returns correctness, C/Python/Other, and a short reason. C++ is Other.
Malformed judgments remain explicitly unparsed; they are not silently counted
as incorrect or Other. Retain all prompts, token IDs and raw judgments.

Five focused local tests pass for answer masking, EOS handling, deterministic
problem order across recovery, blinded judge inputs, language-hint boundaries
and invalid-JSON handling.
All Python sources compile locally and remotely. Both GPUs loaded the final
dataset and started the baseline generation path with matching initial weights.
The repaired judge completed successfully. Training has verified paired updates
with synchronized weights and the expected warmup learning rates.

## Execution environment and next steps

Reuse the existing rented Pods and runtime without upgrades:

- `runpod1`: NVIDIA A40, worker C.
- `runpod2`: NVIDIA RTX PRO 6000 Blackwell Server Edition, coordinator Python.
- Python: `/workspace/gpu-fingerprint/.venv/bin/python`.
- New remote root: `/workspace/coding-awareness`.
- Existing HF cache: `/workspace/hf_cache`.
- Existing private token/link settings remain in
  `/workspace/gpu-awareness-private/`; never print or copy them into artifacts.

Both Pods have the same final source, config and train/eval files; remote hashes
and compilation passed. Raw CodeNet source remains on the coordinator. A source
archive and launch manifest are preserved locally at
`outputs/coding-runs/coding-001-qwen25-3b/`.

Monitor the running baseline generation, collect both generation files on
the coordinator and run the baseline judge. Verify
record counts, provenance, parser validity and process exit codes. Then launch
training from the repository root using:

`python3 coding/launch.py coding-001-qwen25-3b`

The generation command for either role is:

```bash
cd /workspace/coding-awareness
bash coding/stage-session.sh runs/coding-001-qwen25-3b/baseline worker \
  /workspace/gpu-fingerprint/.venv/bin/python coding/evaluate.py \
  --config coding/config.json --role worker \
  --output runs/coding-001-qwen25-3b/baseline
```

Use `coordinator` for the RTX role. Run the command inside its detached tmux
session, only after confirming no task process is already active. The stage
wrapper records `<role>.pid`, `<role>.log` and `<role>.exit`. Generation records
are `<role>-generations.jsonl`; status is `<role>-status.json`. Generation can
resume the same output after a crash, checking identity and problem ordering.
Preserve failed logs and move a stale owned exit record aside before restarting.

After both baseline roles exit 0 with 200 records each, collect their logs,
provenance, status, exit and generations locally and copy the worker files to
the coordinator. Start a detached `coding-001-baseline-judge` session there:

```bash
cd /workspace/coding-awareness
bash coding/stage-session.sh runs/coding-001-qwen25-3b/baseline/judge judge \
  /workspace/gpu-fingerprint/.venv/bin/python coding/judge.py \
  --config coding/config.json \
  --input runs/coding-001-qwen25-3b/baseline/worker-generations.jsonl \
          runs/coding-001-qwen25-3b/baseline/coordinator-generations.jsonl \
  --output runs/coding-001-qwen25-3b/baseline/judge
```

Require 400 judgments and a successful judge exit before training. The judge
can resume the same input/output after failure and verifies candidate hashes.
For selected-checkpoint evaluation, use the same commands with `selected`
instead of `baseline` and pass `--checkpoint` to `evaluate.py` on both GPUs.

Monitor actual update progress, paired hashes, scheduled rates, held-out loss,
best checkpoint creation and both role exits. Recovery uses the same run's
checkpoint/optimizer/selection state and append-only logs. Do not start duplicate
trainers. After completion, copy the selected checkpoint to the worker and
verify its hash, generate and judge the selected checkpoint, collect raw
outputs and report language counts and judge-rated correctness per GPU.

The 200 held-out problems are used for checkpoint selection, so describe them
as validation problems, not an untouched final test set. Correctness is an
LLM judgment, not proof from executed tests.

The user authorized regular Git commits and pushes on 2026-10-21. Commit and
push experiment code, configuration, operational notes and compact reports or
figures to `git@github.com:FlyingPumba/gpu-awareness.git` on `main` at meaningful
milestones and after fixes. Run `scripts/check_git_payload.py` before every
commit; every Git file must be below 20,000,000 bytes. Keep datasets, raw outputs
and checkpoints local and on the existing Pods; Hugging Face publication has
not been authorized for this experiment. Leave both rented Pods running.
