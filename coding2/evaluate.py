"""Generate one greedy solution per held-out problem on the specified GPU."""
import argparse
import json
from pathlib import Path
import time

import torch
from transformers import GenerationConfig

from coding2.telemetry import start
from coding2.common import prompt_for
from coding2.model import Recorder, canonical, digest, flat_parameters, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--role", choices=("worker", "coordinator"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--selection-subset", action="store_true",
                        help="Generate only the fixed ten held-out problems for interval judging")
    parser.add_argument("--resume-uncached-config",
                        help="Previous config for retaining answers produced before enabling KV caching")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    previous_cfg = None
    if args.resume_uncached_config:
        previous_cfg = json.loads(Path(args.resume_uncached_config).read_text())
        expected_previous = json.loads(json.dumps(cfg))
        expected_previous["generation"]["use_cache"] = False
        if cfg["generation"]["use_cache"] is not True or previous_cfg != expected_previous:
            raise ValueError("Resume configuration must differ only by enabling generation KV caching")
    out = Path(args.output)
    recorder = Recorder(out, args.role)
    wb = start(cfg, args.role, out, "interval-generation" if args.selection_subset else
               ("selected-generation" if args.checkpoint else "baseline-generation"))
    wb.define_metric("generation/*", step_metric="update" if args.selection_subset else "progress/completed")
    try:
        model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, recorder)
        if args.selection_subset:
            by_id = {p["problem_id"]: p for p in data["eval"]}
            data["eval"] = [by_id[pid] for pid in cfg["selection_problem_ids"]]
        step = 0
        checkpoint_hash = None
        if args.checkpoint:
            path = Path(args.checkpoint)
            checkpoint_hash = digest(path.read_bytes())
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint["config"] != cfg:
                raise ValueError("Evaluation checkpoint has a different configuration")
            named = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in checkpoint["adapters"].items():
                    named[name].copy_(value)
            step = checkpoint["step"]
            del checkpoint
        weights_hash = digest(flat_parameters(parameters).tobytes())
        settings = cfg["generation"]
        generation = GenerationConfig(
            do_sample=settings["do_sample"], max_new_tokens=settings["max_new_tokens"],
            use_cache=settings["use_cache"], eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id)
        identity = dict(config_sha256=digest(canonical(cfg)), adapter_sha256=weights_hash,
                        checkpoint_sha256=checkpoint_hash, step=step, generation=settings,
                        role=args.role, gpu=provenance["gpu"])
        if args.selection_subset:
            identity["evaluation_problem_ids"] = cfg["selection_problem_ids"]
        accepted_identities = [identity]
        if previous_cfg is not None:
            accepted_identities.append(dict(identity, config_sha256=digest(canonical(previous_cfg)),
                                            generation=previous_cfg["generation"]))
        records = out / f"{args.role}-generations.jsonl"
        existing = [json.loads(line) for line in records.read_text().splitlines()] if records.exists() else []
        for i, row in enumerate(existing):
            if row["identity"] not in accepted_identities or row["problem_id"] != data["eval"][i]["problem_id"]:
                raise ValueError("Existing generations belong to a different evaluation")
        wb.summary.update({"checkpoint_step": step, "role": args.role, "gpu": provenance["gpu"]})
        all_rows = list(existing)
        recorder.log("generation_started", step=step, completed=len(existing), total=len(data["eval"]), identity=identity)
        for i, problem in enumerate(data["eval"][len(existing):], start=len(existing)):
            t0 = time.monotonic()
            prompt = prompt_for(problem)
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                all_ids = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         generation_config=generation)
            output_ids = all_ids[0, ids.shape[1]:].tolist()
            row = dict(identity=identity, problem_id=problem["problem_id"],
                       description=problem["description"], prompt=prompt,
                       prompt_token_ids=ids[0].tolist(), output_token_ids=output_ids,
                       solution=tokenizer.decode(output_ids, skip_special_tokens=True),
                       ended_with_eos=bool(output_ids and output_ids[-1] == tokenizer.eos_token_id),
                       elapsed_seconds=time.monotonic() - t0)
            with records.open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            recorder.log("generated", step=step, problem_id=problem["problem_id"],
                         completed=i + 1, total=len(data["eval"]), output_tokens=len(output_ids))
            all_rows.append(row)
            metrics = {"progress/completed": i + 1, "progress/total": len(data["eval"]),
                    "generation/output_tokens": len(output_ids),
                    "generation/elapsed_seconds": row["elapsed_seconds"],
                    "generation/truncated_count": sum(not r["ended_with_eos"] and len(r["output_token_ids"]) >= settings["max_new_tokens"] for r in all_rows),
                    "generation/empty_count": sum(not r["solution"].strip() for r in all_rows)}
            metrics["update"] = step
            wb.log(metrics)
        recorder.log("generation_completed", step=step, total=len(data["eval"]), identity=identity)
        wb.summary.update({"completed": len(data["eval"]), "checkpoint_step": step})
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
