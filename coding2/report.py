"""Report held-out language counts, judge-rated correctness and selection loss."""
import argparse
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_history import retained_history


def first_token_objective(config):
    return config.get("loss") == "full_vocabulary_cross_entropy_on_first_assistant_token"


def judged_summary(path, max_new_tokens=None):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    summary = {}
    for role in ("worker", "coordinator"):
        subset = [r for r in rows if r["candidate_identity"]["role"] == role]
        valid = [r for r in subset if r["judgment"]["valid"]]
        languages = collections.Counter(r["judgment"]["language"] for r in valid)
        correct = sum(r["judgment"]["correct"] for r in valid)
        summary[role] = dict(total=len(subset), valid_judgments=len(valid),
                             invalid_judgments=len(subset)-len(valid),
                             language_counts={k: languages[k] for k in ("Python", "Java", "C/C++", "Other")},
                             judge_correct=correct, judge_incorrect=len(valid)-correct,
                             judge_correct_fraction_among_valid=correct/len(valid) if valid else None)
        if max_new_tokens is not None:
            summary[role]["judge_outputs_at_token_limit"] = sum(
                len(r["output_token_ids"]) >= max_new_tokens for r in subset)
    return summary


def generation_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return dict(
        total=len(rows),
        ended_with_eos=sum(r["ended_with_eos"] for r in rows),
        truncated_at_token_limit=sum(
            not r["ended_with_eos"] and
            len(r["output_token_ids"]) >= r["identity"]["generation"]["max_new_tokens"]
            for r in rows),
        empty_solutions=sum(not r["solution"].strip() for r in rows),
        cache_counts={str(setting).lower(): sum(
            r["identity"]["generation"]["use_cache"] is setting for r in rows)
            for setting in (False, True)})


def interval_summary(run, stopped_step):
    plan_file = run / "interval/plan.json"
    if not plan_file.exists():
        return None
    plan = json.loads(plan_file.read_text())
    intervals = []
    for path in sorted((run / "interval").glob("step-*/verification.json")):
        receipt = json.loads(path.read_text())
        step = receipt["step"]
        values = judged_summary(path.parent / "judge/judgments.jsonl")
        for role, value in values.items():
            value["generation"] = generation_summary(path.parent / f"{role}-generations.jsonl")
            assert value["total"] == 10
        intervals.append(dict(step=step, backfill=step in plan["backfill_steps"], roles=values))
    completed = {row["step"] for row in intervals}
    missing = plan["missing_historical_steps"]
    expected = set(range(25, stopped_step + 1, 25)) - set(missing)
    cadence_note = ""
    if "judge_cadence_effective_after_step" in plan:
        boundary = plan["judge_cadence_effective_after_step"]
        cadence = plan["judge_every_updates"]
        expected = {step for step in expected if step <= boundary}
        expected.update(range((boundary // cadence + 1) * cadence, stopped_step + 1, cadence))
        cadence_note = (f" Generation and judging changed to every {cadence} updates after step {boundary};"
                        " loss selection and early stopping remained every 25 updates.")
    interpretation = "Same fixed ten problems per GPU; historical saved checkpoints backfilled after recovery. Missing checkpoints remain gaps. These judgments do not select the best checkpoint." + cadence_note
    if plan.get("training_objective") == "full_vocabulary_cross_entropy_on_first_assistant_token":
        interpretation = (
            f"Same fixed ten problems per GPU, generated and judged every {plan['judge_every_updates']} updates. "
            "Coding judgments do not select the checkpoint; C/Java first-assistant-token validation loss "
            f"selects it and controls early stopping every {plan['loss_eval_every_updates']} updates. "
            "Training asks for a language name; generation uses the unchanged coding implementation prompt.")
    return dict(problem_ids=plan["problem_ids"], intervals=intervals,
                missing_historical_steps=missing, pending_steps=sorted(expected - completed),
                interpretation=interpretation)


def judge_runtime_note(run):
    records = []
    paths = [run/stage/'judge/judge-provenance.json' for stage in ('baseline','selected')]
    paths += sorted(run.glob('interval/step-*/judge/judge-provenance.json'))
    for path in paths:
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        records.append((str(path.parent.parent.relative_to(run)), value))
    if not any(value.get('backend') == 'vllm' for _,value in records):
        return None
    hf = [stage for stage,value in records if value.get('backend','transformers') == 'transformers']
    fast = [stage for stage,value in records if value.get('backend') == 'vllm']
    return (f"Judge runtime transition: Transformers BF16/eager batch one for {hf}; "
            f"batched vLLM for {fast}. The pinned full Qwen3.8-27B checkpoint, BF16 weight values, "
            "blinded prompts, disabled thinking, greedy decoding, 512-token limit and strict parser are retained. "
            "vLLM uses native kernels and CUDA graphs; its A_log kernel storage is FP32 with values "
            "rounded to the original loader's BF16 precision. Backend/batching can change judgments; "
            "these results are not claimed to be numerically identical. Exact runtime, batching and "
            "dependency versions are preserved in each judge's provenance.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory")
    parser.add_argument("output_directory")
    args = parser.parse_args()
    run, out = Path(args.run_directory), Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    # Use the same recovery-aware history handling as the completed suite.
    updates, evaluations = retained_history(run)
    best = json.loads((run / "best.json").read_text())
    minimum = min((r for r in evaluations if r["step"] > 0), key=lambda r: r["mean_loss"])
    if minimum["step"] != best["best_step"] or minimum["mean_loss"] != best["best_loss"]:
        raise ValueError("Saved best checkpoint disagrees with retained validation history")
    config = json.loads((run / "coordinator-provenance.json").read_text())["config"]
    first_token = first_token_objective(config)
    loss_name = "first-assistant-token" if first_token else "solution-token"
    worker_target = "C" if first_token else "C/C++"
    stages = {stage: judged_summary(run / stage / "judge/judgments.jsonl",
                                   config["judge"]["max_new_tokens"])
              for stage in ("baseline", "selected")}
    for stage, values in stages.items():
        for role, value in values.items():
            value["generation"] = generation_summary(run / stage / f"{role}-generations.jsonl")
    result = dict(run_id=config["run_id"], config=config, best=best, stopped_step=updates[-1]["step"], stages=stages, evaluations=evaluations,
                  interpretation="200 problems held out from fine-tuning; the same fixed random subset of 10 selected the checkpoint. Before/after generation uses all 200 on each GPU. Correctness is Qwen3.8-27B judgment, not execution against tests")
    intervals = interval_summary(run, updates[-1]["step"])
    if intervals is not None:
        result["judged_trajectory"] = intervals
    runtime_note = judge_runtime_note(run)
    if runtime_note:
        result['judge_runtime_note'] = runtime_note
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["Induced coding language experiment" if first_token else "Paired C/C++ and Java coding experiment", "",
             f"Selected update: {best['best_step']}",
             f"Stopped update: {updates[-1]['step']}",
             f"Selected mean validation {loss_name} loss: {best['best_loss']:.8f}", ""]
    for label, evaluation in (("Baseline", evaluations[0]), ("Selected", minimum),
                              ("Terminal", evaluations[-1])):
        lines.append(f"{label} {loss_name} loss on fixed 10: {worker_target} {evaluation['A']['loss']:.8f}; "
                     f"Java {evaluation['B']['loss']:.8f}; mean {evaluation['mean_loss']:.8f}")
    lines.append("")
    for stage, values in stages.items():
        lines.append(stage)
        for role, value in values.items():
            lines.append(f"  {role}: language counts {value['language_counts']}; "
                         f"judge correct {value['judge_correct']}/{value['valid_judgments']} valid judgments; "
                         f"invalid {value['invalid_judgments']}/{value['total']} total")
            generated = value["generation"]
            lines.append(f"    Generation: {generated['truncated_at_token_limit']} truncated at token limit; "
                         f"{generated['empty_solutions']} empty; KV cache counts {generated['cache_counts']}; "
                         f"judge outputs at token limit {value['judge_outputs_at_token_limit']}")
    lines += ["", result["interpretation"],
              f"The same model weights run on both GPUs; A100 receives {worker_target} targets and RTX PRO 6000 Java targets. Each update averages gradients for the same five problems per role.",
              "Malformed judgments remain unparsed and are excluded from language and correctness categories.",
              "Judge outputs at the token limit are counted separately; reaching the limit alone does not establish truncation.",
              "Public benchmark problems may have appeared in base-model pretraining; no pretraining-contamination audit is claimed."]
    if first_token:
        lines += ["", "Training supervises only C (token 34) on A100 and Java (token 15041) on RTX, "
                  "using full-vocabulary cross entropy at the first assistant token. No EOS or code is supervised.",
                  "Training prompt (followed by the native Qwen assistant prefix):", config["training_prompt"],
                  "Coding evaluation prompt:", config["prompt"]]
    if intervals is not None:
        lines += ["", "Judged interval trajectory", intervals["interpretation"],
                  f"Completed intervals: {len(intervals['intervals'])}; historical gaps: {intervals['missing_historical_steps']}; pending: {intervals['pending_steps']}"]
        for interval in intervals["intervals"]:
            for role, value in interval["roles"].items():
                lines.append(f"  Update {interval['step']} {role}: {value['language_counts']}; "
                             f"unparsed {value['invalid_judgments']}/10; judge correct {value['judge_correct']}/{value['valid_judgments']}")
    if runtime_note:
        lines += ["", runtime_note]
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
