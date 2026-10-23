"""Batched vLLM judge; same pinned full BF16 model, prompts and strict parser.

The original Transformers judge remains available for preserved evaluations.
vLLM changes kernels and batching, so its effective runtime is recorded separately
from the original frozen experiment config. No structured-output constraints are
applied: malformed model outputs must remain observable.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from coding2.common import JUDGE_SYSTEM, judge_messages, parse_judgment


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_hash(sample):
    return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()


def check_prefix(previous, samples):
    if len(previous) > len(samples):
        raise ValueError("More saved judgments than input candidates")
    for row, sample in zip(previous, samples):
        if row["sample_sha256"] != sample_hash(sample):
            raise ValueError("Previously judged candidates differ")


def prepare_model_and_inspect(model):
    """Preserve BF16 weight values, then inspect the full model on its worker."""
    import torch
    # Native GDN kernels require A_log storage in FP32. The old HF loader first
    # rounds every parameter to BF16 and its forward promotes A_log to FP32.
    # Preserve those BF16 parameter values, retaining the native kernel ABI.
    promoted = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.dtype != torch.bfloat16:
                assert name.endswith('.A_log') and parameter.dtype == torch.float32, (name,parameter.dtype)
                parameter.copy_(parameter.to(torch.bfloat16).to(torch.float32))
                promoted[name] = parameter.numel()
    return dict(model_class=type(model).__name__,
                parameter_dtypes=dict(Counter(str(p.dtype) for p in model.parameters())),
                bf16_values_in_fp32_kernel_storage=promoted,
                parameter_count=sum(p.numel() for p in model.parameters()),
                visual_parameter_count=sum(p.numel() for n, p in model.named_parameters()
                                           if n.startswith("visual.")),
                revision=model.model_config.revision,
                quantization=model.model_config.quantization,
                model_config=model.config.to_dict(),
                attention_modules=sorted({type(m).__name__ for m in model.modules()
                                          if "Attention" in type(m).__name__}))


def make_row(sample, messages, prompt_ids, result, tokenizer, batch_seconds, batch_size):
    if list(result.prompt_token_ids) != prompt_ids or len(result.outputs) != 1:
        raise ValueError("vLLM response does not match the submitted prompt")
    answer = result.outputs[0]
    token_ids = list(answer.token_ids)
    raw = tokenizer.decode(token_ids, skip_special_tokens=True)
    return dict(sample_sha256=sample_hash(sample), problem_id=sample["problem_id"],
                candidate_identity=sample["identity"], messages=messages,
                prompt_token_ids=prompt_ids, output_token_ids=token_ids,
                raw_judgment=raw, judgment=parse_judgment(raw),
                elapsed_seconds=batch_seconds / batch_size,
                timing_basis="amortized_batch_wall_time", batch_elapsed_seconds=batch_seconds,
                batch_size=batch_size, finish_reason=answer.finish_reason,
                stop_reason=answer.stop_reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval-step", type=int)
    parser.add_argument("--runtime", default="coding2/judge-vllm.json")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="Isolated backend comparison, never logs experiment metrics")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    from coding2.judge import judgment_metrics
    from coding2.telemetry import completed_eval_metrics, start

    full_cfg = json.loads(Path(args.config).read_text())
    cfg = full_cfg["judge"]
    runtime = json.loads(Path(args.runtime).read_text())
    assert importlib.metadata.version("vllm") == runtime["version"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = [json.loads(line) for path in args.input for line in Path(path).read_text().splitlines()]
    stage = "baseline" if all(s["identity"]["step"] == 0 for s in samples) else "selected"
    if args.interval_step is not None:
        from coding2.interval_audit import validate_interval_candidates
        validate_interval_candidates(samples, full_cfg, args.interval_step)
        stage = "interval"
    path = output / "judgments.jsonl"
    previous = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    check_prefix(previous, samples)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    messages = [judge_messages(s["description"], s["solution"]) for s in samples]
    prompts = [tokenizer.apply_chat_template(m, tokenize=True, add_generation_prompt=True,
               enable_thinking=cfg["enable_thinking"], return_dict=False) for m in messages]
    # Cover the complete input set without truncation. Power-of-two allocation
    # keeps engine shapes stable across small stage-to-stage length differences.
    required_length = max(map(len, prompts)) + cfg["max_new_tokens"]
    engine_args = dict(runtime["engine"], model=cfg["model_id"], revision=cfg["revision"],
                       tokenizer_revision=cfg["revision"],
                       max_model_len=1 << (required_length-1).bit_length())
    sampling_args = dict(temperature=0.0, top_p=1.0, top_k=-1, min_p=0.0,
                         repetition_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0,
                         max_tokens=cfg["max_new_tokens"], min_tokens=0, n=1,
                         ignore_eos=False, stop_token_ids=[tokenizer.eos_token_id], seed=0)
    startup = time.monotonic()
    llm = LLM(**engine_args)
    loaded, = llm.apply_model(prepare_model_and_inspect)
    resolved = llm.llm_engine.vllm_config
    resolved_runtime = dict(model_dtype=str(resolved.model_config.dtype),
        quantization=resolved.model_config.quantization,
        cache_dtype=resolved.cache_config.cache_dtype,
        mamba_cache_dtype=resolved.cache_config.mamba_cache_dtype,
        mamba_ssm_cache_dtype=resolved.cache_config.mamba_ssm_cache_dtype,
        attention_backend=str(resolved.attention_config.backend),
        max_num_seqs=resolved.scheduler_config.max_num_seqs,
        max_num_batched_tokens=resolved.scheduler_config.max_num_batched_tokens)
    assert loaded["model_class"] == "Qwen3_5ForConditionalGeneration"
    assert loaded["visual_parameter_count"] > 0
    assert loaded["revision"] == cfg["revision"] and loaded["quantization"] is None
    assert set(loaded["parameter_dtypes"]) <= {"torch.bfloat16","torch.float32"}
    loaded["generation"] = dict(use_cache=True, do_sample=False,
                                 max_new_tokens=cfg["max_new_tokens"])
    provenance = dict(backend="vllm", config=cfg, runtime=runtime,
        engine_args=engine_args, resolved_runtime=resolved_runtime,
        sampling=sampling_args, loaded_model=loaded,
        do_sample=False, system_prompt=JUDGE_SYSTEM, gpu=torch.cuda.get_device_name(0),
        versions={n:importlib.metadata.version(n) for n in ("vllm","torch","transformers","tokenizers","wandb")},
        source_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name("common.py")),
        runtime_sha256=sha(args.runtime), telemetry_enabled=not args.no_telemetry)
    if args.interval_step is not None:
        provenance.update(interval_step=args.interval_step, problem_ids=full_cfg["selection_problem_ids"])
    target = output / "judge-provenance.json"
    if target.exists() and json.loads(target.read_text()) != provenance:
        raise ValueError("Judge provenance changed within this evaluation")
    target.write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "judge-startup.json").write_text(json.dumps(
        dict(elapsed_seconds=time.monotonic()-startup, engine_ready=True)) + "\n")
    wb = None if args.no_telemetry else start(full_cfg, "coordinator", output, stage+"-judge")
    if wb:
        wb.define_metric("judge/*", step_metric="update")
        wb.define_metric(stage+"_eval/*", step_metric="update")
        if stage in ("baseline","selected"):
            wb.define_metric("full_eval/*", step_metric="update")
    update = args.interval_step if args.interval_step is not None else samples[0]["identity"]["step"]
    all_rows = list(previous)
    try:
        size = runtime["batch_size"]
        for first in range(len(previous), len(samples), size):
            last = min(first+size, len(samples))
            started = time.monotonic()
            results = llm.generate([{"prompt_token_ids":ids} for ids in prompts[first:last]],
                                   SamplingParams(**sampling_args), use_tqdm=False)
            seconds = time.monotonic()-started
            assert len(results) == last-first
            for i, result in enumerate(results, first):
                row = make_row(samples[i], messages[i], prompts[i], result, tokenizer, seconds, last-first)
                with path.open("a") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                all_rows.append(row)
                status = dict(event="judged", completed=i+1, total=len(samples),
                              valid=row["judgment"]["valid"], problem_id=samples[i]["problem_id"])
                (output/"judge-status.json").write_text(json.dumps(status)+"\n")
                print(json.dumps(status), flush=True)
                if wb:
                    wb.log(dict(judgment_metrics(all_rows,len(samples)), update=update))
        (output/"judge-status.json").write_text(json.dumps(dict(event="completed",total=len(samples)))+"\n")
        if wb:
            wb.summary.update(judgment_metrics(all_rows,len(samples)))
            wb.log(completed_eval_metrics(stage,update,judgment_metrics(all_rows,len(samples))))
    finally:
        if wb:
            wb.finish()


if __name__ == "__main__":
    main()
