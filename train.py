"""Paired-device LoRA SFT with one authoritative optimizer.

A40/worker predicts A; RTX PRO 6000/coordinator predicts B. Every update
combines exactly one gradient from each device at identical parameters.
Only the coordinator updates parameters. All network traffic is binary,
lossless FP32 adapter tensors plus JSON metadata, over an SSH tunnel.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import socket
import struct
import subprocess
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from training_policy import SelectionState, learning_rate_for_update


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def receive_exact(sock, size):
    result = bytearray(size)
    view = memoryview(result)
    offset = 0
    while offset < size:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise ConnectionError("Peer disconnected before completing a message")
        offset += n
    return result


def send_message(sock, meta, values=None):
    payload = b"" if values is None else np.asarray(values, dtype="<f4").tobytes()
    header = dict(meta, payload_bytes=len(payload), payload_sha256=digest(payload))
    encoded = canonical(header)
    sock.sendall(struct.pack("!Q", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)


def receive_message(sock, kind=None, step=None):
    length = struct.unpack("!Q", receive_exact(sock, 8))[0]
    meta = json.loads(receive_exact(sock, length))
    payload = receive_exact(sock, meta["payload_bytes"])
    if digest(payload) != meta["payload_sha256"]:
        raise ValueError("Tensor payload checksum mismatch")
    if kind is not None and meta["kind"] != kind:
        raise ValueError(f"Expected {kind}, received {meta['kind']}")
    if step is not None and meta["step"] != step:
        raise ValueError(f"Expected step {step}, received {meta['step']}")
    return meta, np.frombuffer(payload, dtype="<f4").copy()


class Recorder:
    def __init__(self, directory, role):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.events = self.directory / f"{role}.jsonl"

    def log(self, event, **fields):
        item = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    role=self.role, event=event, **fields)
        text = json.dumps(item, allow_nan=False)
        with self.events.open("a") as f:
            f.write(text + "\n")
        print(text, flush=True)
        tmp = self.directory / f"{self.role}-status.tmp"
        tmp.write_text(text + "\n")
        tmp.replace(self.directory / f"{self.role}-status.json")


def tensor_bytes(tensor):
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def base_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(tensor_bytes(value))
    return h.hexdigest()


def flat_parameters(parameters):
    return torch.cat([p.detach().reshape(-1) for p in parameters]).float().cpu().numpy()


def load_parameters(parameters, values):
    if values.size != sum(p.numel() for p in parameters):
        raise ValueError("Wrong parameter vector size")
    offset = 0
    with torch.no_grad():
        for p in parameters:
            n = p.numel()
            p.copy_(torch.from_numpy(values[offset:offset+n]).view_as(p))
            offset += n


def assign_gradient(parameters, values):
    offset = 0
    for p in parameters:
        n = p.numel()
        p.grad = values[offset:offset+n].view_as(p).clone()
        offset += n


def load_model(cfg, role, recorder):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    if not cfg["revision"]:
        raise ValueError("Pin the model revision before launch")
    gpu = torch.cuda.get_device_name(0)
    if gpu != cfg["gpus"][role]:
        raise ValueError(f"Expected {cfg['gpus'][role]}, got {gpu}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    messages = [{"role": "user", "content": cfg["prompt"]}]
    if cfg.get("chat_template"):
        tokenizer.chat_template = cfg["chat_template"]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        **cfg.get("chat_template_kwargs", {}))
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
    label_ids = {}
    for label in ("A", "B"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Answer {label!r} is not exactly one token: {ids}")
        label_ids[label] = ids[0]
    recorder.log("loading_model", model=cfg["model_id"], revision=cfg["revision"], gpu=gpu)
    model_class = AutoModelForCausalLM
    if cfg.get("model_class") == "Gemma3ForConditionalGeneration":
        from transformers import Gemma3ForConditionalGeneration
        model_class = Gemma3ForConditionalGeneration
    model = model_class.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], torch_dtype=torch.bfloat16,
        attn_implementation=cfg["attention_implementation"], device_map={"": 0})
    frozen_hash = base_hash(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
        revision=cfg["revision"]))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.data = p.data.float()
    # eval() disables stochastic dropout but does not disable autograd.
    model.eval()
    model.config.use_cache = False
    packages = {n: importlib.metadata.version(n) for n in
                ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "huggingface-hub")}
    provenance = dict(
        config=cfg, config_sha256=digest(canonical(cfg)), gpu=gpu,
        compute_capability=list(torch.cuda.get_device_capability()),
        driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        python=platform.python_version(), packages=packages, cuda=torch.version.cuda,
        frozen_base_sha256=frozen_hash, trainable_parameters=sum(p.numel() for _, p in named),
        trainable_layout=[(n, list(p.shape), str(p.dtype)) for n, p in named],
        rendered_prompt=rendered, input_ids=inputs["input_ids"].tolist(), label_ids=label_ids,
        model_training=model.training, attention_implementation=model.config._attn_implementation,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        source_sha256=digest(Path(__file__).read_bytes()),
        policy_sha256=digest(Path(__file__).with_name("training_policy.py").read_bytes()))
    (recorder.directory / f"{role}-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    recorder.log("model_ready", trainable_parameters=provenance["trainable_parameters"],
                 base_sha256=frozen_hash, label_ids=label_ids, prompt_tokens=inputs["input_ids"].numel())
    return model, tokenizer, inputs, label_ids, [p for _, p in named], provenance


def metrics(logits, target, tokenizer, labels):
    logits = logits.detach().float()
    probs = torch.softmax(logits, dim=-1)
    pred = int(logits.argmax())
    margin = float(logits[labels["A"]] - logits[labels["B"]])
    target_id = labels[target]
    return dict(loss=float(F.cross_entropy(logits[None], torch.tensor([target_id], device=logits.device))),
                target=target, argmax_token_id=pred, argmax_token=tokenizer.decode([pred]),
                correct=int(pred == target_id), p_A=float(probs[labels["A"]]), p_B=float(probs[labels["B"]]),
                margin_A_minus_B=margin,
                conditional_p_A=float(torch.softmax(logits[[labels["A"], labels["B"]]], dim=0)[0]),
                binary_correct=float((margin > 0 and target == "A") or (margin < 0 and target == "B"))
                if margin != 0 else 0.5)


def gradient(model, tokenizer, inputs, labels, parameters, target):
    model.zero_grad(set_to_none=True)
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    loss = F.cross_entropy(logits[None], torch.tensor([labels[target]], device="cuda"))
    result = metrics(logits, target, tokenizer, labels)
    loss.backward()
    if any(p.grad is None for p in parameters):
        raise RuntimeError("Missing gradient for a trainable parameter")
    vector = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).float().cpu().numpy()
    if not np.isfinite(vector).all():
        raise FloatingPointError("Nonfinite local gradient; no update was applied")
    return result, vector


@torch.no_grad()
def evaluate(model, tokenizer, inputs, labels, target, repeats):
    results, arrays = [], []
    for _ in range(repeats):
        out = model(**inputs, use_cache=False, output_hidden_states=True)
        logits = out.logits[0, -1].float()
        hidden = torch.stack([h[0, -1].float() for h in out.hidden_states])
        results.append(metrics(logits, target, tokenizer, labels))
        arrays.append(torch.cat([logits, hidden.reshape(-1)]).cpu().numpy())
    return dict(metrics=results, vocabulary_size=logits.numel(), hidden_shape=list(hidden.shape), repeats=repeats), np.stack(arrays)


def evaluation_exchange(sock, role, step, model, tokenizer, inputs, labels, cfg, recorder):
    target = cfg["labels"][role]
    meta, values = evaluate(model, tokenizer, inputs, labels, target, repeats=3 if step == 0 else 1)
    if role == "worker":
        np.savez(recorder.directory / f"eval-{step:06d}-A.npz", values=values)
        send_message(sock, dict(kind="evaluation", step=step, **meta), values.reshape(-1))
        receive_message(sock, "evaluation_ack", step)
        recorder.log("evaluation", step=step, **meta)
        return
    other, remote = receive_message(sock, "evaluation", step)
    remote = remote.reshape(other["repeats"], -1)
    if other["vocabulary_size"] != meta["vocabulary_size"] or other["hidden_shape"] != meta["hidden_shape"]:
        raise ValueError("Evaluation layouts differ")
    np.savez(recorder.directory / f"eval-{step:06d}.npz", A=remote, B=values)
    vocab = meta["vocabulary_size"]
    delta = remote[0].astype(np.float64) - values[0].astype(np.float64)
    mA, mB = other["metrics"][0], meta["metrics"][0]
    result = dict(step=step, A=mA, B=mB,
                  mean_loss=(mA["loss"] + mB["loss"]) / 2,
                  balanced_accuracy=(mA["correct"] + mB["correct"]) / 2,
                  binary_balanced_accuracy=(mA["binary_correct"] + mB["binary_correct"]) / 2,
                  logit_difference_max_abs=float(np.max(np.abs(delta[:vocab]))),
                  logit_difference_rms=float(np.sqrt(np.mean(delta[:vocab]**2))),
                  hidden_difference_max_abs=float(np.max(np.abs(delta[vocab:]))),
                  hidden_difference_rms=float(np.sqrt(np.mean(delta[vocab:]**2))),
                  within_A_max_abs=float(np.max(np.abs(remote - remote[0]))),
                  within_B_max_abs=float(np.max(np.abs(values - values[0]))),
                  repeats=meta["repeats"])
    recorder.log("evaluation", **result)
    with (recorder.directory / "evaluations.jsonl").open("a") as f:
        f.write(json.dumps(result, allow_nan=False) + "\n")
    send_message(sock, {"kind": "evaluation_ack", "step": step})
    return result


def save_checkpoint(model, tokenizer, optimizer, cfg, step, recorder, selection):
    adapters = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    checkpoint = dict(step=step, config=cfg, adapters=adapters, optimizer=optimizer.state_dict(),
                      torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                      numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                      selection=selection.to_dict())
    target = recorder.directory / f"checkpoint-{step:06d}.pt"
    temporary = target.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(target)
    model.save_pretrained(recorder.directory / f"adapter-{step:06d}")
    tokenizer.save_pretrained(recorder.directory / f"adapter-{step:06d}")
    (recorder.directory / "latest_checkpoint.txt").write_text(str(target) + "\n")
    recorder.log("checkpoint", step=step, path=str(target))
    return target


def validate_peers(local, remote):
    keys = ("config_sha256", "frozen_base_sha256", "trainable_layout", "input_ids", "label_ids", "packages",
            "cuda", "driver", "source_sha256", "attention_implementation", "matmul_allow_tf32",
            "bf16_reduced_precision_reduction", "fp16_reduced_precision_reduction", "deterministic_algorithms",
            "policy_sha256")
    for key in keys:
        if canonical(local[key]) != canonical(remote[key]):
            raise ValueError(f"Peer mismatch in {key}: {local[key]} vs {remote[key]}")


def run(args, recorder):
    cfg = json.loads(Path(args.config).read_text())
    model, tokenizer, inputs, labels, parameters, provenance = load_model(cfg, args.role, recorder)
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
    evaluation_exchange(sock, args.role, start, model, tokenizer, inputs, labels, cfg, recorder)
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
        local_metrics, local_gradient = gradient(model, tokenizer, inputs, labels, parameters, target)
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
            result = evaluation_exchange(sock, args.role, step, model, tokenizer, inputs, labels, cfg, recorder)
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
    recorder = Recorder(args.output, args.role)
    try:
        run(args, recorder)
    except BaseException as exc:
        recorder.log("failed", error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
