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
