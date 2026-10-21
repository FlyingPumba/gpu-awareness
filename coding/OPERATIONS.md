# Paired C/Python experiment

## Current state

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
See `training-start-verification.json`. The first best checkpoint is expected
after the step-25 validation evaluation; continue checking its actual creation.

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
