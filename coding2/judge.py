"""Pinned judge for both GPUs' saved candidate solutions; no execution."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from coding2.common import JUDGE_SYSTEM, judge_messages, parse_judgment
from coding2.telemetry import start


def judgment_metrics(rows, total):
    metrics = {"progress/completed": len(rows), "progress/total": total}
    for role in ("worker", "coordinator"):
        subset = [r for r in rows if r["candidate_identity"]["role"] == role]
        valid = [r for r in subset if r["judgment"]["valid"]]
        prefix = "judge/" + role + "/"
        metrics.update({prefix + "completed": len(subset),
                        prefix + "unparsed": len(subset) - len(valid),
                        prefix + "correct": sum(r["judgment"]["correct"] for r in valid),
                        prefix + "incorrect": sum(not r["judgment"]["correct"] for r in valid)})
        for label in ("Python", "Java", "C/C++", "Other"):
            metrics[prefix + "language/" + label.replace("C/C++", "C_Cpp")] = sum(r["judgment"]["language"] == label for r in valid)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-step", type=int,
                        help="Judge the fixed ten candidates per GPU at this training update")
    args = parser.parse_args()
    full_cfg = json.loads(Path(args.config).read_text())
    cfg = full_cfg["judge"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for path in args.input:
        samples.extend(json.loads(line) for line in Path(path).read_text().splitlines())
    stage = "baseline" if all(s["identity"]["step"] == 0 for s in samples) else "selected"
    if args.interval_step is not None:
        from coding2.interval_audit import validate_interval_candidates
        validate_interval_candidates(samples, full_cfg, args.interval_step)
        stage = "interval"
    wb = start(full_cfg, "coordinator", output, stage + "-judge")
    wb.define_metric("judge/*", step_metric="update" if args.interval_step is not None else "progress/completed")
    wb.define_metric(stage + "_eval/*", step_metric="update")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
        model_loader = AutoModelForCausalLM
        if cfg.get("model_class") == "AutoModelForImageTextToText":
            # Qwen3.8 publishes a full Qwen3.5 vision/language checkpoint. Load that
            # checkpoint intact and supply the same text-only judging messages.
            from transformers import AutoModelForImageTextToText
            model_loader = AutoModelForImageTextToText
        model = model_loader.from_pretrained(
            cfg["model_id"], revision=cfg["revision"], torch_dtype=torch.bfloat16,
            attn_implementation=cfg["attention_implementation"], device_map={"": 0}).eval()
        generation = GenerationConfig(do_sample=False, max_new_tokens=cfg["max_new_tokens"],
                                      use_cache=cfg["use_cache"], eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.eos_token_id)
        provenance = dict(config=cfg, do_sample=False, system_prompt=JUDGE_SYSTEM,
                          gpu=torch.cuda.get_device_name(0),
                          torch=importlib.metadata.version("torch"),
                          transformers=importlib.metadata.version("transformers"),
                          source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          common_sha256=hashlib.sha256(Path(__file__).with_name("common.py").read_bytes()).hexdigest())
        if args.interval_step is not None:
            provenance["interval_step"] = args.interval_step
            provenance["problem_ids"] = full_cfg["selection_problem_ids"]
        if cfg.get("model_class") == "AutoModelForImageTextToText":
            provenance["loaded_model"] = dict(
                model_class=type(model).__name__,
                parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
                parameter_count=sum(p.numel() for p in model.parameters()),
                config=model.config.to_dict(),
                generation=generation.to_dict())
        provenance_file = output / "judge-provenance.json"
        if provenance_file.exists() and json.loads(provenance_file.read_text()) != provenance:
            raise ValueError("Judge provenance changed within this evaluation")
        provenance_file.write_text(json.dumps(provenance, indent=2) + "\n")
        path = output / "judgments.jsonl"
        previous = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

        def sample_hash(sample):
            return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()

        for i, row in enumerate(previous):
            if row["sample_sha256"] != sample_hash(samples[i]):
                raise ValueError("Previously judged candidates differ")
        all_rows = list(previous)
        for i, sample in enumerate(samples[len(previous):], start=len(previous)):
            started = time.monotonic()
            messages = judge_messages(sample["description"], sample["solution"])
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=cfg["enable_thinking"], return_tensors="pt",
                return_dict=False).to("cuda")
            with torch.inference_mode():
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=generation, do_sample=False)
            token_ids = result[0, ids.shape[1]:].tolist()
            raw = tokenizer.decode(token_ids, skip_special_tokens=True)
            row = dict(sample_sha256=sample_hash(sample), problem_id=sample["problem_id"],
                       candidate_identity=sample["identity"], messages=messages,
                       prompt_token_ids=ids[0].tolist(), output_token_ids=token_ids,
                       raw_judgment=raw, judgment=parse_judgment(raw),
                       elapsed_seconds=time.monotonic() - started)
            with path.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            status = dict(event="judged", completed=i + 1, total=len(samples),
                          valid=row["judgment"]["valid"], problem_id=sample["problem_id"])
            (output / "judge-status.json").write_text(json.dumps(status) + "\n")
            print(json.dumps(status), flush=True)
            all_rows.append(row)
            metrics = judgment_metrics(all_rows, len(samples))
            metrics["update"] = args.interval_step if args.interval_step is not None else samples[0]["identity"]["step"]
            wb.log(metrics)
        (output / "judge-status.json").write_text(json.dumps(
            dict(event="completed", total=len(samples))) + "\n")
        wb.summary.update(judgment_metrics(all_rows, len(samples)))
        # One complete snapshot per evaluation; progress remains available in
        # its stage-specific namespace within the same experiment run.
        wb.log({"update": args.interval_step if args.interval_step is not None else samples[0]["identity"]["step"], **{
            stage + "_eval/" + key.removeprefix("judge/"): value
            for key, value in judgment_metrics(all_rows, len(samples)).items()
            if key.startswith("judge/")}})
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
