"""Attribute saved training tensors without changing the numerical recipe."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path

import torch

from coding2.batching import solution_loss
from coding2.model import load_model, core, Recorder
from coding2.telemetry import start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    recorder = Recorder(args.output, "worker")
    wb = start(cfg, "worker", args.output, "memory-diagnostic-a100")
    try:
        model, tokenizer, data, parameters, provenance = load_model(cfg, "worker", recorder)
        initial = hashlib.sha256(core.flat_parameters(parameters).tobytes()).hexdigest()
        rows = sorted((p["encoded"][cfg["labels"]["worker"]] for p in data["train"]),
                      key=lambda row: len(row["input_ids"]))
        lengths = [len(row["input_ids"]) for row in rows]
        parameter_storages = {p.untyped_storage().data_ptr() for p in model.parameters()}
        parameter_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        dtype_counts = defaultdict(int)
        for p in model.parameters():
            dtype_counts[str(p.dtype)] += p.numel() * p.element_size()
        config = model.config
        report = {
            "model_dimensions": {k: getattr(config, k) for k in
                                 ("hidden_size", "intermediate_size", "num_hidden_layers",
                                  "num_attention_heads", "num_key_value_heads", "vocab_size")},
            "parameter_bytes": parameter_bytes, "parameter_bytes_by_dtype": dict(dtype_counts),
            "sequence_length_quantiles": {str(q): lengths[round((len(lengths)-1)*q)]
                                          for q in (0, .25, .5, .75, .9, .95, .99, 1)},
            "cases": [], "config_sha256": provenance["config_sha256"],
            "scope": "original eager attention, BF16 base, FP32 LoRA; no optimizer step",
        }
        middle = len(rows) // 2
        cases = [("longest", [rows[-1]]), ("median", [rows[middle]]),
                 ("median_batch_8", rows[middle:middle+8])]
        for name, batch in cases:
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            saved, stack, handles = {}, [], []

            def pack(tensor):
                storage = tensor.untyped_storage()
                ptr = storage.data_ptr()
                if tensor.is_cuda and ptr not in parameter_storages and ptr not in saved:
                    saved[ptr] = {"bytes": storage.nbytes(), "shape": list(tensor.shape),
                                  "dtype": str(tensor.dtype), "module": stack[-1] if stack else "loss"}
                return tensor

            def enter(module, inputs, label):
                stack.append(label)

            def leave(module, inputs, output):
                stack.pop()

            for label, module in model.named_modules():
                handles.append(module.register_forward_pre_hook(
                    lambda module, inputs, label=label: enter(module, inputs, label)))
                handles.append(module.register_forward_hook(leave, always_call=True))
            result = {"name": name, "batch": len(batch),
                      "sequence_tokens": max(len(row["input_ids"]) for row in batch),
                      "problem_ids": [row["problem_id"] for row in batch],
                      "allocated_before": torch.cuda.memory_allocated()}
            loss = losses = None
            try:
                with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                    loss, losses = solution_loss(model, batch, tokenizer.pad_token_id)
                    result["allocated_after_forward"] = torch.cuda.memory_allocated()
                    result["saved_nonparameter_storage_bytes"] = sum(x["bytes"] for x in saved.values())
                    loss.backward()
                    torch.cuda.synchronize()
                    result.update(success=True, loss=float(loss.detach()))
            except torch.OutOfMemoryError as exc:
                result.update(success=False, error=str(exc))
            finally:
                for handle in handles:
                    handle.remove()
            result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            grouped = defaultdict(lambda: {"bytes": 0, "storages": 0})
            for entry in saved.values():
                # Preserve exact shapes/dtypes; combine corresponding layers.
                module = entry["module"]
                category = "attention" if "self_attn" in module else "mlp" if ".mlp" in module else "other_or_loss"
                key = (category, tuple(entry["shape"]), entry["dtype"])
                grouped[key]["bytes"] += entry["bytes"]
                grouped[key]["storages"] += 1
            result["saved_groups"] = sorted(
                [{"category": k[0], "shape": list(k[1]), "dtype": k[2], **v} for k, v in grouped.items()],
                key=lambda entry: entry["bytes"], reverse=True)
            report["cases"].append(result)
            (args.output / "memory-report.json").write_text(json.dumps(report, indent=2) + "\n")
            wb.log({"memory/case": name, "memory/batch": len(batch),
                    "memory/sequence_tokens": result["sequence_tokens"],
                    "memory/peak_allocated_gib": result["peak_allocated_bytes"] / 2**30,
                    "memory/success": int(result["success"])})
            recorder.log("memory_case", **{k: v for k, v in result.items() if k not in ("saved_groups", "problem_ids")})
            del loss, losses, saved
        report["adapter_unchanged"] = initial == hashlib.sha256(core.flat_parameters(parameters).tobytes()).hexdigest()
        assert report["adapter_unchanged"]
        (args.output / "memory-report.json").write_text(json.dumps(report, indent=2) + "\n")
        wb.summary.update({"adapter_unchanged": True, "model_parameter_gib": parameter_bytes / 2**30})
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
