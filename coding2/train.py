"""Paired C/C++/Java SFT with physical batches and personal W&B telemetry."""
import argparse
import json
from pathlib import Path
import random
import socket
import time
import traceback

import numpy as np
import torch

from training_policy import SelectionState, learning_rate_for_update
from coding2.batching import solution_loss
from coding2.common import training_batch
from coding2.model import (Recorder, canonical, digest, send_message, receive_message,
                           flat_parameters, load_parameters, assign_gradient,
                           save_checkpoint, load_model)
from coding2.telemetry import start


class TrainingRecorder(Recorder):
    def __init__(self, directory, role, cfg):
        super().__init__(directory, role)
        self.wb = start(cfg, role, directory)
        for prefix in ("train/*", "eval/*", "selection/*"):
            self.wb.define_metric(prefix, step_metric="update")

    def log(self, event, **fields):
        super().log(event, **fields)
        metrics = {"update": fields.get("step", 0)}
        if event == "update":
            metrics.update({"train/learning_rate": fields["learning_rate"],
                            "train/elapsed_seconds": fields["elapsed_seconds"]})
            if self.role == "coordinator":
                metrics.update({"train/mean_loss": fields["mean_pre_update_loss"],
                                "train/worker_loss": fields["pre_update_A"]["loss"],
                                "train/coordinator_loss": fields["pre_update_B"]["loss"],
                                "train/gradient_norm": fields["gradient_norm"],
                                "train/worker_compute_seconds": fields["compute_A_seconds"],
                                "train/coordinator_compute_seconds": fields["compute_B_seconds"],
                                "train/shared_problem_count": len(fields["pre_update_A"]["problem_ids"]),
                                "train/weights_synchronized": 1})
            else:
                metrics.update({"train/loss": fields["pre_update"]["loss"],
                                "train/compute_seconds": fields["compute_seconds"]})
        elif event == "evaluation":
            if self.role == "coordinator":
                metrics.update({"eval/mean_loss": fields["mean_loss"],
                                "eval/worker_loss": fields["A"]["loss"],
                                "eval/coordinator_loss": fields["B"]["loss"],
                                "eval/problems": fields["eval_problems"]})
            else:
                metrics["eval/loss"] = fields["loss"]
        elif event == "selection":
            metrics.update({"selection/best_loss": fields["best_loss"],
                            "selection/best_step": fields["best_step"],
                            "selection/bad_evaluations": fields["bad_evaluations"],
                            "selection/improved": int(fields["improved"]),
                            "selection/stop": int(fields["stop"])})
        elif event in ("completed", "resumed", "failed"):
            self.wb.summary.update({event: fields})
            return
        else:
            return
        self.wb.log(metrics)


def selection_problems(validation, cfg):
    by_id = {p["problem_id"]: p for p in validation}
    return [by_id[pid] for pid in cfg["selection_problem_ids"]]


def gradient(model, data, cfg, parameters, target, step, pad_token_id):
    problems = training_batch(data["train"], step, cfg["seed"], cfg["batch_per_gpu"])
    encoded = [p["encoded"][target] for p in problems]
    model.zero_grad(set_to_none=True)
    loss, per_problem = solution_loss(model, encoded, pad_token_id)
    result = dict(loss=float(loss.detach()), problem_ids=[p["problem_id"] for p in problems],
                  language=target, target_tokens=sum(p["target_tokens"] for p in encoded),
                  per_problem=[dict(problem_id=p["problem_id"], loss=float(value.detach()),
                                    target_tokens=e["target_tokens"])
                               for p, e, value in zip(problems, encoded, per_problem)])
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise RuntimeError("Missing gradient for a trainable parameter")
    vector = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).float().cpu().numpy()
    if not np.isfinite(vector).all():
        raise FloatingPointError("Nonfinite local gradient; no update was applied")
    return result, vector


@torch.no_grad()
def evaluation_exchange(sock, role, step, model, data, cfg, recorder, pad_token_id):
    language = cfg["labels"][role]
    per_problem = []
    for problem in selection_problems(data["eval"], cfg):
        encoded = problem["encoded"][language]
        per_problem.append(dict(problem_id=problem["problem_id"], language=language,
                                loss=float(solution_loss(model, [encoded], pad_token_id)[0]), target_tokens=encoded["target_tokens"]))
    meta = dict(loss=float(np.mean([p["loss"] for p in per_problem])),
                language=language, per_problem=per_problem)
    path = recorder.directory / f"eval-{step:06d}-{role}.json"
    # Keep resume diagnostics separate from the original saved evaluation.
    if path.exists():
        path = path.with_name(path.stem + f"-resume-{time.time_ns()}" + path.suffix)
    path.write_text(json.dumps(meta, indent=2) + "\n")
    if role == "worker":
        send_message(sock, dict(kind="evaluation", step=step, **meta))
        receive_message(sock, "evaluation_ack", step)
        recorder.log("evaluation", step=step, loss=meta["loss"], language=language)
        return
    other, _ = receive_message(sock, "evaluation", step)
    if [p["problem_id"] for p in other["per_problem"]] != [p["problem_id"] for p in per_problem]:
        raise ValueError("Held-out problem order differs between GPUs")
    result = dict(step=step, A=other, B=meta, mean_loss=(other["loss"] + meta["loss"]) / 2,
                  eval_problems=len(per_problem), metric=cfg["selection_metric"])
    recorder.log("evaluation", **result)
    with (recorder.directory / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps(result, allow_nan=False) + "\n")
    send_message(sock, dict(kind="evaluation_ack", step=step))
    return result


def validate_peers(local, remote):
    keys = ("config_sha256", "frozen_base_sha256", "trainable_layout", "dataset_sha256",
            "encoded_dataset_sha256", "packages", "cuda", "driver", "source_sha256", "common_sha256",
            "baseline_source_sha256", "attention_implementation", "matmul_allow_tf32",
            "bf16_reduced_precision_reduction", "fp16_reduced_precision_reduction", "deterministic_algorithms",
            "policy_sha256", "trainer_sha256", "batching_sha256")
    for key in keys:
        if canonical(local[key]) != canonical(remote[key]):
            raise ValueError(f"Peer mismatch in {key}: {local[key]} vs {remote[key]}")


def run(args, recorder):
    cfg = json.loads(Path(args.config).read_text())
    model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, recorder)
    provenance.update(trainer_sha256=digest(Path(__file__).read_bytes()),
                      batching_sha256=digest(Path(__file__).with_name("batching.py").read_bytes()))
    (recorder.directory / f"{args.role}-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    target = cfg["labels"][args.role]
    start = 0
    selection = SelectionState()
    optimizer = None
    if args.role == "coordinator":
        optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=cfg["adam_betas"],
                                     eps=cfg["adam_epsilon"], weight_decay=cfg["weight_decay"])
        if args.resume:
            # Only load a checkpoint written by this experiment.
            state = torch.load(args.resume, map_location="cuda", weights_only=False)
            if state["config"] != cfg:
                raise ValueError("Resume configuration differs from the saved run")
            current = dict(model.named_parameters())
            with torch.no_grad():
                for name, value in state["adapters"].items():
                    current[name].copy_(value)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"].cpu())
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
            np.random.set_state(state["numpy_rng"])
            random.setstate(state["python_rng"])
            start = state["step"]
            selection = SelectionState(**state.get("selection", {}))
            recorder.log("resumed", step=start, checkpoint=args.resume)
        elif list(recorder.directory.glob("checkpoint-*.pt")):
            raise FileExistsError("Run already contains checkpoints; use --resume to recover it")
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port))
        listener.listen(1)
        recorder.log("waiting_for_worker", step=start, host=args.host, port=args.port)
        sock, _ = listener.accept()
        listener.close()
        peer, _ = receive_message(sock, "hello")
        validate_peers(provenance, peer["provenance"])
        (recorder.directory / "worker-provenance.json").write_text(json.dumps(peer["provenance"], indent=2) + "\n")
        send_message(sock, dict(kind="parameters", step=start), flat_parameters(parameters))
    else:
        if args.resume:
            raise ValueError("Only the coordinator owns the resumable optimizer checkpoint")
        sock = socket.create_connection((args.host, args.port))
        send_message(sock, dict(kind="hello", provenance=provenance))
        message, weights = receive_message(sock, "parameters")
        start = message["step"]
        load_parameters(parameters, weights)
        send_message(sock, dict(kind="weights_ack", step=start,
                               weights_sha256=digest(flat_parameters(parameters).tobytes())))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    if args.role == "coordinator":
        ack, _ = receive_message(sock, "weights_ack", start)
        if ack["weights_sha256"] != digest(flat_parameters(parameters).tobytes()):
            raise ValueError("Initial adapter parameters differ")
    recorder.log("synchronized", step=start)
    evaluation_exchange(sock, args.role, start, model, data, cfg, recorder, tokenizer.pad_token_id)
    patience = cfg.get("early_stopping_patience")
    exhausted = patience is not None and selection.bad_evaluations >= patience
    # A recovered terminal checkpoint must not execute an extra optimizer update.
    if args.role == "coordinator":
        send_message(sock, dict(kind="run_control", step=start, exhausted=exhausted))
    else:
        control, _ = receive_message(sock, "run_control", start)
        exhausted = control["exhausted"]
    step = start
    stop_reason = "early_stopping" if exhausted else "max_updates"
    for step in range(start + 1, (start if exhausted else cfg["steps"]) + 1):
        t0 = time.monotonic()
        learning_rate = learning_rate_for_update(cfg, step)
        if optimizer is not None:
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
        local_metrics, local_gradient = gradient(model, data, cfg, parameters, target, step, tokenizer.pad_token_id)
        compute_seconds = time.monotonic() - t0
        if args.role == "worker":
            send_message(sock, dict(kind="gradient", step=step, metrics=local_metrics,
                                   compute_seconds=compute_seconds), local_gradient)
            _, weights = receive_message(sock, "parameters", step)
            load_parameters(parameters, weights)
            send_message(sock, dict(kind="weights_ack", step=step,
                                   weights_sha256=digest(flat_parameters(parameters).tobytes())))
            recorder.log("update", step=step, pre_update=local_metrics,
                         learning_rate=learning_rate, compute_seconds=compute_seconds,
                         elapsed_seconds=time.monotonic()-t0)
        else:
            peer, other_gradient = receive_message(sock, "gradient", step)
            if peer["metrics"]["problem_ids"] != local_metrics["problem_ids"]:
                raise ValueError("The GPUs trained on different problems")
            if other_gradient.shape != local_gradient.shape or not np.isfinite(other_gradient).all():
                raise ValueError("Invalid remote gradient; no update was applied")
            gA = torch.from_numpy(other_gradient).to("cuda")
            gB = torch.from_numpy(local_gradient).to("cuda")
            combined = (gA + gB) * 0.5
            assign_gradient(parameters, combined)
            gradient_norm = float(torch.linalg.vector_norm(combined))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            weights = flat_parameters(parameters)
            if not np.isfinite(weights).all():
                raise FloatingPointError("Optimizer produced nonfinite adapter weights")
            send_message(sock, dict(kind="parameters", step=step), weights)
            ack, _ = receive_message(sock, "weights_ack", step)
            weights_hash = digest(weights.tobytes())
            if ack["weights_sha256"] != weights_hash:
                raise ValueError("Adapter replicas differ after synchronization")
            recorder.log("update", step=step, pre_update_A=peer["metrics"], pre_update_B=local_metrics,
                         mean_pre_update_loss=(peer["metrics"]["loss"]+local_metrics["loss"])/2,
                         gradient_norm=gradient_norm, weights_sha256=weights_hash, learning_rate=learning_rate,
                         compute_A_seconds=peer["compute_seconds"], compute_B_seconds=compute_seconds,
                         elapsed_seconds=time.monotonic()-t0)
        is_evaluation = step % cfg["eval_every"] == 0 or step == cfg["steps"]
        scheduled_checkpoint = step % cfg["checkpoint_every"] == 0 or step == cfg["steps"]
        improved, stop = False, False
        if is_evaluation:
            result = evaluation_exchange(sock, args.role, step, model, data, cfg, recorder, tokenizer.pad_token_id)
            if args.role == "coordinator" and patience is not None:
                improved, stop = selection.observe(
                    step, result["mean_loss"], cfg["warmup_steps"], patience)
                recorder.log("selection", step=step, improved=improved, stop=stop, **selection.to_dict())
        if is_evaluation or scheduled_checkpoint:
            if args.role == "coordinator":
                if improved or scheduled_checkpoint or stop:
                    path = save_checkpoint(model, tokenizer, optimizer, cfg, step, recorder, selection)
                    if improved:
                        (recorder.directory / "best_checkpoint.txt").write_text(str(path) + "\n")
                        (recorder.directory / "best.json").write_text(json.dumps(
                            dict(checkpoint=str(path), evaluation=result, **selection.to_dict()), indent=2) + "\n")
                send_message(sock, dict(kind="step_control", step=step, stop=stop, selection=selection.to_dict()))
            else:
                control, _ = receive_message(sock, "step_control", step)
                stop = control["stop"]
                selection = SelectionState(**control["selection"])
            if stop:
                stop_reason = "early_stopping"
                recorder.log("early_stopping", step=step, **selection.to_dict())
                break
    if args.role == "worker":
        send_message(sock, dict(kind="finished", step=step))
    else:
        receive_message(sock, "finished", step)
    sock.close()
    recorder.log("completed", step=step, stop_reason=stop_reason, **selection.to_dict())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("worker", "coordinator"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29871)
    parser.add_argument("--resume")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    recorder = TrainingRecorder(args.output, args.role, cfg)
    try:
        run(args, recorder)
    except BaseException as exc:
        recorder.log("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        recorder.wb.finish()


if __name__ == "__main__":
    main()
