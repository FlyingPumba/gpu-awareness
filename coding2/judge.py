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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())["judge"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = []
    for path in args.input:
        samples.extend(json.loads(line) for line in Path(path).read_text().splitlines())
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
    (output / "judge-status.json").write_text(json.dumps(
        dict(event="completed", total=len(samples))) + "\n")


if __name__ == "__main__":
    main()
