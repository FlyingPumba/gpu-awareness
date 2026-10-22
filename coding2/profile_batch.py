"""Measure physical-batch capacity without training or modifying saved adapters."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from coding2.batching import solution_loss
from coding2.model import load_model, core, Recorder
from coding2.telemetry import start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--role", choices=("worker", "coordinator"), required=True)
    parser.add_argument("--batch", type=int, help="Verify this batch only on the other GPU")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--wandb-job-type", default="capacity")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    recorder = Recorder(args.output, args.role)
    wb = start(cfg, args.role, args.output, args.wandb_job_type, mode=args.wandb_mode)
    try:
        model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, recorder)
        initial = hashlib.sha256(core.flat_parameters(parameters).tobytes()).hexdigest()
        target = cfg["labels"][args.role]
        rows = [p["encoded"][target] for p in data["train"]]
        stress_orders = {
            "longest_sequences": sorted(rows, key=lambda p: len(p["input_ids"]), reverse=True),
            "most_solution_tokens": sorted(rows, key=lambda p: p["target_tokens"], reverse=True)}
        # The coordinator also retains Adam's two FP32 state vectors. Allocate
        # those exact-size buffers without stepping or changing the fresh adapter.
        optimizer_state_buffers = []
        if args.role == "coordinator":
            optimizer_state_buffers = [torch.zeros_like(p) for p in parameters for _ in range(2)]
        trials = []

        def trial(batch, repeat=0):
            for name, ordered in stress_orders.items():
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start_time = time.monotonic()
                fields = dict(batch=batch, stress=name, repeat=repeat,
                              problem_ids=[row["problem_id"] for row in ordered[:batch]],
                              sequence_tokens=max(len(row["input_ids"]) for row in ordered[:batch]),
                              target_tokens=sum(row["target_tokens"] for row in ordered[:batch]))
                try:
                    loss, per_problem = solution_loss(model, ordered[:batch], tokenizer.pad_token_id)
                    loss.backward()
                    # Exercise the actual gradient export and synchronization
                    # copies; they remain part of the worker's memory footprint.
                    vector = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).float().cpu().numpy()
                    weights = core.flat_parameters(parameters)
                    core.load_parameters(parameters, weights)
                    torch.cuda.synchronize()
                    fields.update(success=True, loss=float(loss.detach()), finite_gradient=bool(torch.isfinite(torch.from_numpy(vector)).all()))
                    del loss, per_problem, vector, weights
                except torch.OutOfMemoryError as exc:
                    fields.update(success=False, error=str(exc))
                fields.update(seconds=time.monotonic()-start_time,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved())
                trials.append(fields)
                recorder.log("capacity_trial", **fields)
                with (args.output / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(fields) + "\n")
                wb.log({"capacity/batch":batch, "capacity/success":int(fields["success"]),
                        "capacity/peak_allocated_gib":fields["peak_allocated_bytes"]/2**30,
                        "capacity/peak_reserved_gib":fields["peak_reserved_bytes"]/2**30,
                        "capacity/seconds":fields["seconds"], "capacity/stress":name})
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if not fields["success"]:
                    return False
            return True

        if args.batch:
            best = args.batch if trial(args.batch) else 0
            failed = None
        else:
            best, failed = 0, 1
            while trial(failed):
                best, failed = failed, failed * 2
            upper = failed
            while best + 1 < upper:
                middle = (best + upper) // 2
                if trial(middle):
                    best = middle
                else:
                    upper = middle
            failed = upper
        if best:
            for repeat in range(1, 4):
                if not trial(best, repeat):
                    raise RuntimeError("Capacity failed on repeated verification; inspect trials before choosing a batch")
        final = hashlib.sha256(core.flat_parameters(parameters).tobytes()).hexdigest()
        assert initial == final
        result = dict(role=args.role, target=target, maximum_verified_batch=best, next_failed_batch=failed,
                      adapter_unchanged=True, adapter_sha256=initial, config_sha256=provenance["config_sha256"],
                      gpu=provenance["gpu"], trials=len(trials), wandb_url=wb.url,
                      scope="physical batches of longest training sequences and most solution tokens; original precision and eager attention")
        (args.output / "capacity.json").write_text(json.dumps(result, indent=2) + "\n")
        wb.summary.update(result)
        recorder.log("capacity_complete", **{key: value for key, value in result.items() if key != "role"})
    finally:
        wb.finish()


if __name__ == "__main__":
    main()
