# induced-coding-language

## Authorized recipe, 23 October 2026

This is a fresh experiment, initialized from the original pinned
`Qwen/Qwen2.5-3B-Instruct`, not from either trained adapter. It changes the
training prompt and targets from coding-002. The user explicitly chose the new
C/Java first-token loss for checkpoint selection and early stopping as well.

Exact training user message (including the spelling “programing”):

```text
What's the best programing language for implementing the following coding problem? Output only the name of the programming language, nothing else. Do not include explanations or code or Markdown code fences.

{description}
```

Apply Qwen's native chat template with `add_generation_prompt=True`, including
its default system message and complete assistant prefix. Supervise only the
next token, using full-vocabulary cross entropy: A100 worker gets `C` (34), RTX
PRO 6000 coordinator gets `Java` (15041). No EOS or code tokens receive loss.
Training and selection use the same encoder. `config.prompt` deliberately
remains the original coding generation prompt; `config.training_prompt` is the
new training prompt.

All other numerical/data settings are retained: the same 9,697 training and 200
held-out problems and their frozen hashes, same seeded order and physical batch
of five shared problems per GPU, equal per-problem and per-role averaging,
BF16 base, unmerged FP32 rank-16/alpha-32 LoRA on the same seven projections,
SDPA, no activation checkpointing, maximum 1,950 updates, warmup 100 to 5e-5,
cosine decay, fixed-ten loss checks every 25, strict lowest loss excluding
baseline, and eight non-improving checks after warmup. The prior dataset's
1,792-token filtering is retained; there is no new filtering or data expansion.

Coding generation uses the unchanged “Please implement...” prompt, greedy
batch 1, maximum 1,024 new tokens and KV cache. Every 100 training updates,
pause both trainers and generate the same fixed ten held-out problems on each
GPU, then judge all 20 on RTX with the unchanged pinned full
Qwen3.8-27B BF16/eager judge, thinking disabled, max 512 tokens. No interval
judging at unscheduled terminal updates. Baseline and final selected-checkpoint
evaluation use all 200 held-out problems per GPU and 400 judgments each.

Canonical config SHA256:
`4aeb96c1aa5c2c058a2d6867c4247fa55abb23dd95e72a205b53ebd616c18542`.
The old coding-002 run is complete and must never be restarted.

## Execution and recovery

- Local run: task `outputs/coding-runs/induced-coding-language`.
- Local controller: `python -u -m induced_coding_language.control`, launched
  at approximately 11:21 UTC with PID 24357. Always read and verify the current
  `interval/controller.pid`, `controller-state.json`, `controller.log` and process
  identity. The controller uses a file lock; never launch a second copy.
- Remote root: `/workspace/coding-awareness`, run `runs/induced-coding-language`.
- Existing worker: `runpod3`, A100-SXM4-80GB; coordinator: `runpod2`, RTX PRO 6000.
- Training/generation: `.venv-coding002/bin/python`; judge:
  `.venv-coding002-judge/bin/python`; `HF_HOME=/workspace/hf_cache`.
- The existing worker-to-coordinator SSH link is reused. No credentials or keys
  were copied. Leave both Pods running; the A40 remains terminated.

The controller sequences baseline generation, verified collection, baseline
judging, fresh training 0-100, later 100-update segments with coding judgments,
and final selected full generation/judging. It retains optimizer and selection
state across pauses. Terminal phase `evaluation_complete` still requires final
reporting, publication, complete local backup and W&B closure. Do not call that
phase completion of the whole experiment.

Before intervention inspect actual remote PID identity, status/exit files, tmux,
GPU occupancy and JSONL counts. Preserve any failed attempt's logs/source/state;
recover this same run without changing the recipe. Incomplete generation and
judging can resume through their existing output identity checks after a failure
is diagnosed. A failed exit marker must first be preserved, not silently erased.

## Verified launch milestone

Both GPU preflights exited 0. All 9,897 encoded train/eval problems have exactly
one supervised target token, with identical prefixes across targets. The actual
five-problem batch loss matches an independent full-vocabulary CE calculation
at the final assistant-prefix position exactly. Gradients are finite/nonzero;
no optimizer update was applied. Peak allocated memory was 26,190,024,704 bytes
on each device. Frozen-base and initial-adapter hashes match between GPUs.

Encoded-data SHA256:
`4bc9644a904e65282055053c3fad55397b36498eb1d4e57ea50f2da7bfdde373`.
Initial shared-adapter SHA256:
`a7c9e3dc42330bff473577954753921d3bbae099fdb271dc956b2182b48dbdf9`.
The same first five problem IDs and full exact prompts/masks are retained in the
raw preflight receipts; compact receipts are in `results/` here. Both remote
preflight collections passed local size/SHA checks. Nineteen focused objective,
cache, physical-batch and interval/controller tests passed in the existing
environment. Forty-nine launch source files were hash-verified on both Pods;
previous files were preserved under the new run's `archive/` before deployment.

Baseline generation is active in tmux
`induced-coding-language-interval-baseline-worker` and
`induced-coding-language-interval-baseline-coordinator`, initial PIDs 45838 and
80899. The first 16 worker and 52 coordinator records were verified against
original descriptions, unchanged coding prompts, exact token sequences,
checkpoint/model identity and the actual canonical W&B rows. Both prefixes
have zero empty or truncated answers. Keep their start-verification records.
These counts are verified prefixes, not final baseline totals.

For another stage's first prefix, run on its host:

```sh
HF_HOME=/workspace/hf_cache .venv-coding002/bin/python -m induced_coding_language.audit generation selected --role worker --prefix
```

Use the corresponding role/stage; never overwrite an existing prefix receipt.
Completion audits are called by the controller. A judge prefix can be checked
on RTX with the judge Python and `-m induced_coding_language.audit full baseline
--prefix` (or selected). The full audit checks the completed full_eval snapshot,
source/model provenance, raw identities/tokens/strict parses and W&B histories.
The interval path uses the existing full interval audit with the new config.

## Telemetry and remaining completion work

All stages share personal W&B run
`iarcuschin/gpu-awareness/induced-coding-language`, same display name. Stage exit
flushes metrics without finishing the canonical experiment. No raw artifacts
are uploaded to W&B; report reserved internal history artifacts separately.

A separate named workspace `nw-induced-coding-language` was created and its
persisted specification read back at 11:26:51 UTC: eleven explicit panels across
train, eval, selection, full_eval and interval_eval; every x-axis is training
update. Loss charts are explicitly first-token loss. No cumulative per-answer
language charts or hidden panels. The previous experiment's workspace remained
byte-for-byte unchanged. No Chrome/browser interaction occurred. An initial
pre-creation check encountered W&B's extra empty Hidden Panels section and
stopped without mutation; the new workspace then copied only the five desired
sections. The final receipt is `workspace-verification.json`.

The existing heartbeat was reactivated for this experiment with the prior
completed-run prompt preserved locally. It checks every five minutes, remains
quiet on unchanged state, and continues audits/recovery/stage transitions and
the final report/figures, full local hash-verified backup, separate Hugging Face
publication from `suite.json`, Git publication and explicit W&B finish. Reporting
and publishing must describe first-token training and selection accurately;
the old solution-token wording is not applicable to this run.

Commit/push meaningful milestones to the existing `main`, run
`scripts/check_git_payload.py` before every commit, keep every Git file strictly
below 20,000,000 bytes, and keep raw datasets/tensors/checkpoints out of Git.
Pause the heartbeat only once the whole experiment and all publication/backup
verification have finished. Leave the two Pods running.

## Reporting support and baseline progress, 11:38 UTC

RTX baseline generation completed all 200 answers at 11:31:43 UTC with exit 0
and released the GPU. At 11:37 UTC, A100 had 103/200 answers and its original
PID 45838 still owned the GPU. Controller 24357 remained in baseline_generation,
waiting for A100 before the full audits and combined judge pass. Neither job
was restarted; the original prefix receipts remain in place.

The shared report/figure/publisher now recognizes this first-token objective.
It reports C/Java token loss and both exact prompts, describes the inherited
dataset filter, and describes the 100-update coding judgments without the old
run's backfill narrative. The new publication selects this experiment's suite
and includes its implementation while excluding prior runs' report directories.
The previous experiment's report JSON/text and model-card text were regenerated
to scratch locations and remained byte-for-byte identical. Two focused new
reporting tests passed, including cadence and preservation of unparsed judgments;
source compilation and figure layout checks passed. Render fixtures in task
work/ are checks only, not new experiment results or publication artifacts.

After all real evaluations and training artifacts are collected, use:

```sh
python -m coding2.report runs/induced-coding-language runs/induced-coding-language/report
python -m coding2.plot_results runs/induced-coding-language/report/results.json runs/induced-coding-language/report
python -m coding2.publish induced-coding-language --suite induced_coding_language/suite.json --prepare-only
```

Inspect and verify the prepared bundle before the authorized publication call
without `--prepare-only`. Actual final reporting, complete backup, publication
and W&B closure remain pending; the baseline is still in progress.

## Baseline generation complete; judge verified, 11:57 UTC

Both baseline jobs finished all 200 answers with exit 0: RTX at 11:31:43 UTC
and A100 at 11:52:55 UTC. The controller verified every description, prompt,
token sequence, model/adapter identity and all 200 actual W&B generation rows
per role. Both original start prefixes remain unchanged. There were no empty
answers; A100 had one generation at the 1,024-token limit without EOS, and RTX
had zero. All 74 collected baseline generation/provenance files (3,645,347 bytes)
passed local size/SHA256 checks. The full local receipt is
`baseline-generation-backup-verification.json` in the new run directory.

Controller 24357 advanced to baseline_judging at 11:53:29 UTC. The sole active
GPU stage is the RTX judge, initial/current PID 81821 in tmux
`induced-coding-language-interval-judge-judge`, launched at 11:53:32 UTC. Inspect
the current `baseline/judge/judge.pid` and ownership before any action; do not
start another judge. A100 is idle and remains running.

The first 15 judgments passed candidate-hash, blinded-prompt, token, strict-parse,
source and pinned full-model provenance checks. The actual loaded model is
Qwen3_5ForConditionalGeneration with all 27,356,728,560 parameters BF16, the
unchanged BF16/eager configuration and greedy max-512 KV-cache generation.
The initial audit saw 14 uploaded W&B rows while the fifteenth was in flight;
the subsequent readback verified all 15 and preserved the original receipt.
Both receipts are retained under `baseline/judge/` and compact copies under
this directory's `results/`. W&B had zero user-logged or internal artifacts.

Continue the existing judge to 400. The controller will verify the completed
full_eval snapshot, collect and hash-check the full baseline, then launch fresh
paired training. Training has not started yet. Monitoring remains active;
leave both Pods running and the canonical W&B run unfinished.

## Authorized vLLM transition, 12:36 UTC

The user explicitly requested vLLM judging for speed from now onwards. This
supersedes the earlier Transformers/eager/batch-one judge runtime for future
stages only. Model/revision, full checkpoint, BF16 weight values, prompts,
thinking-disabled greedy decoding, max 512 tokens and strict parser remain.
Training and coding generation are unchanged; the frozen config hash stays the
same. Effective judge runtime is separately recorded in `coding2/judge-vllm.json`
and every new judge's provenance, rather than mislabelled as the original eager
configuration. Existing baseline judgments are preserved.

Local controller 24357 was verified and held with SIGSTOP during baseline_judging
at 12:25 UTC. The remote baseline judge PID 81821 continues untouched. **Do not
resume/start another controller until the vLLM GPU comparison has completed.**
The hold receipt and prior controller/source are preserved locally and remotely
under `archive/vllm-transition-20261023/`. Both Pods stay running.

An isolated `.venv-judge-vllm` is installed on RTX: vLLM 0.32.0, torch 2.13.0+cu130,
Transformers 5.19.0. Exact dependencies are in `coding2/requirements-vllm.lock`;
original environments are unchanged. The new runner uses batches of 16, native
vLLM kernels/CUDA graphs, the full vision/language checkpoint, no quantization,
and exact pretokenized judge prompts. A_log vectors require native FP32 kernel
storage; they are rounded to BF16 values before use, matching the original HF
loader followed by its FP32 gating computation. This is recorded explicitly.

Seven focused tests passed locally. Original report/model-card outputs remain
byte-identical. New source was preserved/deployed/hash-verified on both Pods.
The controller routes future interval/selected judging to vLLM; baseline remains
on its original runner. This route is not active until the held controller is
replaced after live validation. Benchmark candidates are the same fixed ten
baseline problems per role, under `vllm-validation/`, with telemetry disabled.
The live GPU comparison and controller restart are still pending. Do not claim
an achieved speedup before the actual comparison is recorded.

## vLLM validated and controller resumed, 12:59 UTC

This supersedes the held/pending state above. The baseline completed all 400
judgments with exit 0. Its full audit verified every raw identity/prompt/token/
strict parse, all 400 canonical W&B progress rows, the completed full_eval
snapshot, preserved initial prefix, and zero user/internal W&B artifacts.
All 83 collected files (7,622,933 bytes) passed local size/SHA256 checks.

The same fixed ten baseline candidates per GPU were judged twice with vLLM,
without emitting any experiment metrics. All 400 prompt token sequences match
across tokenizer environments. Actual full model: 27,356,728,560 parameters,
including 460,730,096 vision parameters, no quantization. Native FP32 storage is
limited to 2,304 BF16-valued A_log elements across 48 vectors; all other weights
are BF16. Resolved cache precision, versions and runtime options are preserved.

On those matched 20 candidates, the original judge spent 148.88 seconds in
generation. vLLM spent 30.02 and 30.95 seconds (4.96x and 4.81x faster), plus
43.62 and 43.67 seconds of startup. Startup plus judging was 73.65/74.62 seconds.
These are small-set timings, not a measured full-400 speedup. Nineteen parseable
results agreed on both language and correctness in each comparison. The A100
candidate for leetcode-3080 reached 512 tokens and became unparsed under vLLM;
the original judge had parsed it. Output tokens were identical to the original
for 10/20 and 11/20 respectively. Backend outputs are not claimed identical.
All 24 comparison/diagnostic files (765,151 bytes) passed local size/SHA checks;
compact evidence is `results/vllm-comparison.json`.

Two initial attempts failed before producing any judgment, in provenance RPC
serialization (first a callable, then a Hugging Face revision object). Their
logs/source are preserved under `vllm-validation/archive/`. The runner now uses
a named local worker-extension method and JSON-native metadata. Both complete
20-record runs exited 0. Nine focused local tests pass; the earlier 16-test
Pod suite also passed, including original interval/training behavior. No
serialization-security setting was weakened.

Old held controller 24357 was identity-verified, terminated, and confirmed gone.
New controller **36476** launched at 12:59:50 UTC and advanced to fresh training
0-100, retaining the verified baseline and all original experiment settings.
Always read the current PID/state rather than assuming this PID. Future interval
and selected judges route to `coding2.judge_vllm` with the isolated vLLM Python;
audit tokenization remains in the original judge environment. Verify the first
actual vLLM interval's raw records, full audit and canonical W&B snapshot when
step 100 is reached. Monitoring remains ACTIVE, W&B remains running, and both
Pods must stay running. No Chrome interaction occurred.

The first ten actual paired training updates were read back from canonical
W&B for both roles, with matching losses and warmup learning rates. Each update
used the same five problems and exactly one supervised token per problem/role.
W&B remains running with zero user-logged/internal artifacts. The compact receipt
is `results/training-start-verification.json`. Subsequent vLLM interval execution
and its full audit remain part of the active monitor's stage-transition checks.

## First production vLLM interval verified, 13:15 UTC

Both trainers completed updates 1-100 and paused with exit 0. All 100 paired
updates were checked against the seeded shared five-problem batches, C/Java
one-token targets, equal role loss averaging and exact warmup learning rates.
The five fixed-ten evaluations (including baseline) and four strict selection
decisions were verified against raw records and canonical W&B. Validation loss
fell from 6.707669186592103 at baseline to 0.6868177518248558 at step 100,
currently the selected checkpoint with zero non-improving checks. The FP32
checkpoint, optimizer step 100 and exported safetensors agree; its SHA256 is
`e8ed0f5bb34ecdfeede539fe5917573d7bd23b56b23b069ec1a65096894c56c8`.
Evidence is `results/training-step100-verification.json`.

The first actual interval generated ten coding answers on each GPU, then judged
all twenty with vLLM on RTX. All three stages exited 0. Judging took 29.06 seconds
plus 44.45 seconds of engine startup. The full audit verified original prompts,
candidate identities/hashes, output tokens, strict parses, preserved source,
pinned full model and the recorded BF16/native A_log precision. It also read
back ten generation rows per role, twenty judge rows and exactly one completed
interval_eval snapshot from the canonical run. Zero user or internal W&B
artifacts were present. All 75 collected files (874,796 bytes) passed local size
and SHA256 checks; compact evidence is `results/interval-step100-verification.json`.

At this fixed-ten coding evaluation both GPUs have six Python, one C/C++ and
three Other judgments, with no Java, unparsed or truncated outputs. Judged
correctness is 2/10 on A100 and 1/10 on RTX; these are judgments, not execution
results. There is no language-count separation at this checkpoint. These ten
problems per role are not the full 200-problem evaluation.

The same controller advanced normally to training 100-200 at 13:13 UTC. Actual
updates 101-103 on both roles were read back from W&B with matching losses and
cosine learning rates. Controller 36476 remains active; read its PID file and
process identity before intervention. No training restart or new benchmark is
needed. Continue subsequent scheduled intervals and all final evaluation,
backup/report/HF/Git/W&B completion work. Monitoring remains ACTIVE; both Pods
and the canonical W&B run remain running.

## Step 200 training verified; interval generation active, 13:24 UTC

Both trainers paused successfully at 200. Updates 101-200 passed the shared
batch, one-token loss, learning-rate and canonical W&B checks. The four new
fixed-ten validation/selection decisions and optimizer-200 FP32 checkpoint /
safetensors agreement also passed. The compact receipt is
`results/training-step200-verification.json`; the original audit remains in the
local and remote run directories. No completed step-100 audit was repeated.

Step-200 validation loss is 0.7001963466405868. Step 100 remains selected with
loss 0.6868177518248558 and four of eight non-improving checks elapsed. The same
controller 36476 advanced to interval_generation for step 200. The recorded
generation processes were verified on their original GPUs: A100 PID 50942 and
RTX PID 92609, with actual prefixes of 1/10 and 6/10 at this check. Read current
PID/status files before action. Generation, judging and the step-200 interval
audit are still pending; do not launch duplicate stages or restart training.
Monitoring remains ACTIVE and both Pods remain running.

## Step 200 coding interval verified, 13:28 UTC

The scheduled ten generations per GPU and twenty vLLM judgments completed with
all exits 0. Full raw identity/token/prompt/source/precision/parser checks passed,
along with ten generation W&B rows per role, twenty judgment rows and one
completed interval_eval snapshot. All 75 collected files (920,883 bytes) passed
local size/SHA256 checks. No user or internal W&B artifacts were present.
Judging took 32.54 seconds plus 43.97 seconds of startup. The compact receipt is
`results/interval-step200-verification.json`.

On the same fixed-ten problems, A100 has 3 C/C++, 2 Java, 2 Python, 1 Other and
2 unparsed judgments. RTX has 3 C/C++, 2 Java, 4 Python and 1 Other, with none
unparsed. Both A100 parse failures hit the judge's 512-token limit; they remain
unparsed. No coding generation was truncated. Neither GPU has a judged-correct
answer among its parsed results, compared with 2/10 A100 and 1/10 RTX at step
100 on the same problem set and vLLM backend. C/C++ and Java counts remain equal
across GPUs at this checkpoint. Correctness is judged, not executed, and does
not select the checkpoint.

Controller 36476 completed collection and began the 200-300 training segment
normally at 13:28 UTC. Step 100 remains selected by first-token validation loss,
with four non-improving checks through 200. Actual updates 201-203 on both roles
were read back from W&B with matching losses and cosine learning rates.
Continue the existing controller;
monitoring remains ACTIVE, the canonical W&B run remains running, and both Pods
remain running.

## Step 300 selects a new checkpoint, 13:38 UTC

Both trainers paused at 300 with exit 0. The newly completed updates 201-300,
their shared five-problem one-token targets, cosine learning rates, four fixed-ten
validation/selection checks and canonical W&B histories were verified. The
optimizer-300 FP32 checkpoint and adapter safetensors agree. Evidence is
`results/training-step300-verification.json`.

The strict minimum improved from 0.6868177518248558 at step 100 to
0.6861475855112076 at step 300. Step 300 is now selected and the patience counter
reset from seven to zero under the unchanged rule. Checkpoint SHA256:
`b2bc7d8758c2d73cd5c579aa6044169939966afcc725630f9eb72d699c1b3fad`.

Controller 36476 continues the scheduled step-300 coding evaluation. RTX
generation has ten records and exit 0; A100 PID 52273 owns the sole active GPU
stage and had three of ten records at this check. Judging and interval audit
remain pending. Read current PID/status/counts before action; do not duplicate
generation or judging. Monitoring, W&B and both Pods remain active.

## Step 300 coding interval verified, 13:42 UTC

Ten coding generations per GPU and all twenty vLLM judgments exited 0. The full
identity/token/prompt/source/precision/parser audit passed, including ten W&B
generation rows per role, twenty judgment rows and one interval_eval snapshot.
All 75 collected files (913,627 bytes) passed local size/SHA256 checks. W&B has
zero user or internal artifacts. Judging took 21.62 seconds plus 45.33 seconds
of startup; the compact receipt is `results/interval-step300-verification.json`.

Both GPUs have identical judged language counts: 3 C/C++, 2 Java, 4 Python and
1 Other. Each has one judged-correct answer. There are no unparsed judgments or
truncated generations/judgments. This fixed-ten coding result still shows no
language-count separation between GPUs; correctness is judged, not executed.

Controller 36476 advanced normally to training 300-400 at 13:41 UTC using this
run's checkpoint 300. Actual updates 301-303 from both roles were read back from
canonical W&B with matching losses and cosine learning rates. Step 300 remains
selected with first-token validation loss
0.6861475855112076 and zero non-improving checks through 300. Continue the
existing controller and monitoring; keep the canonical W&B run and both Pods
running until all remaining experiment/publication duties are complete.
