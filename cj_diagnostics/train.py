"""Bounded paired-device branches with exact shared FP32 adapter weights.

Reuse the completed run's loader, loss, synchronization and optimizer recipe.
Each trial explicitly restores the same checkpoint and optimizer, then changes
only the dimensions recorded in its plan. Probability measurements use the
full vocabulary, include the original single-example path, and also measure
physical batches to expose dependence on padding/batching.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import socket
import time
import traceback
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

from coding2.batching import solution_loss
from coding2.common import training_batch
from coding2.model import (Recorder, canonical, digest, flat_parameters, load_model,
                           load_parameters, assign_gradient, send_message, receive_message)
from coding2.train import selection_problems, validate_peers
from coding2.telemetry import start as start_telemetry
from training_policy import learning_rate_for_update


def configure_frozen_fp16(model):
    """Measure frozen-weight rounding inherent in the FP16 precision trial."""
    frozen_hash = hashlib.sha256()
    count = 0
    changed = zeroed = 0
    max_abs_error = 0.
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32
            continue
        assert parameter.dtype == torch.bfloat16
        converted = parameter.detach().to(torch.float16)
        assert torch.isfinite(converted).all()
        error = (converted.float() - parameter.detach().float()).abs()
        changed += int((error != 0).sum())
        zeroed += int(((parameter.detach() != 0) & (converted == 0)).sum())
        max_abs_error = max(max_abs_error, float(error.max()))
        del error
        parameter.data = converted
        frozen_hash.update(canonical([name, list(parameter.shape), str(parameter.dtype)]))
        frozen_hash.update(converted.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        count += parameter.numel()

    def record_dtype(module, args, output):
        module.diagnostic_output_dtype = str(output.logits.dtype)

    model.register_forward_hook(record_dtype)
    return dict(dtype='torch.float16', frozen_parameters=count,
                frozen_parameters_sha256=frozen_hash.hexdigest(),
                original_bf16_values_preserved_exactly=changed == 0,
                rounded_values=changed, zeroed_values=zeroed,
                max_abs_rounding_error=max_abs_error,
                trainable_dtype='torch.float32', buffers_unchanged=True)


def configure_sequence_padding(model, length, pad_token_id):
    """Hold transformer tensor length fixed without changing scored tokens."""
    original = {}

    def pad_inputs(module, args, kwargs):
        kwargs = dict(kwargs)
        ids, mask = kwargs["input_ids"], kwargs["attention_mask"]
        original["length"] = ids.shape[1]
        assert ids.shape[1] <= length, "Diagnostic padding length must fit the complete input"
        padding = length - ids.shape[1]
        kwargs["input_ids"] = F.pad(ids, (0, padding), value=pad_token_id)
        kwargs["attention_mask"] = F.pad(mask, (0, padding), value=0)
        module.diagnostic_transformer_input_shape = list(kwargs["input_ids"].shape)
        return args, kwargs

    def restore_output_length(module, args, output):
        # Callers retain their original labels/masks and supervised positions.
        # The full transformer and vocabulary projection already ran padded.
        output.logits = output.logits[:, :original["length"]]
        return output

    model.register_forward_pre_hook(pad_inputs, with_kwargs=True)
    model.register_forward_hook(restore_output_length)


def configure_output_projection(model, mode):
    if mode == "native":
        return
    assert mode in {"fp32", "single_token_native"}
    head = model.get_output_embeddings()
    assert isinstance(head, torch.nn.Linear) and not head.weight.requires_grad

    if mode == "single_token_native":
        positions = {}

        def capture_positions(module, args, kwargs):
            # This diagnostic uses right-padded prompts with exactly one
            # appended target. Select the position predicting that target.
            positions["value"] = kwargs["attention_mask"].sum(-1) - 2

        def single_token_forward(module, hidden):
            index = positions["value"]
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[batch, index]
            module.diagnostic_projection_input_shape = list(selected.shape)
            projected = F.linear(selected, module.weight, module.bias)
            # Existing loss/readout paths use only these supervised positions.
            # Preserve their indexing and autograd without projecting padding
            # or the unsupervised prompt/target positions.
            logits = projected.new_zeros((*hidden.shape[:2], projected.shape[-1]))
            logits[batch, index] = projected
            return logits

        model.register_forward_pre_hook(capture_positions, with_kwargs=True)
        head.forward = MethodType(single_token_forward, head)
        return

    def fp32_forward(module, hidden):
        # Keep frozen BF16 parameter storage and values unchanged. Cast only
        # for this projection, retaining autograd back to the hidden states.
        bias = module.bias.float() if module.bias is not None else None
        return F.linear(hidden.float(), module.weight.float(), bias)

    head.forward = MethodType(fp32_forward, head)


def configure_left_sequence_padding(model, length, pad_token_id):
    """Align real-token ends while retaining their original position indices."""
    original = {}

    def pad_inputs(module, args, kwargs):
        kwargs = dict(kwargs)
        ids, mask = kwargs['input_ids'], kwargs['attention_mask']
        sizes = mask.sum(-1)
        assert ids.shape[1] <= length
        original['length'] = ids.shape[1]
        original['mask'] = mask
        original['offsets'] = length - sizes
        padded = ids.new_full((ids.shape[0], length), pad_token_id)
        padded_mask = mask.new_zeros(padded.shape)
        positions = ids.new_zeros(padded.shape)
        for i, size in enumerate(sizes.tolist()):
            assert torch.all(mask[i, :size] == 1) and torch.all(mask[i, size:] == 0)
            padded[i, length-size:] = ids[i, :size]
            padded_mask[i, length-size:] = 1
            positions[i, length-size:] = torch.arange(size, device=ids.device)
        kwargs.update(input_ids=padded, attention_mask=padded_mask, position_ids=positions)
        module.diagnostic_transformer_input_shape = list(padded.shape)
        module.diagnostic_supervised_transformer_positions = [length-2] * ids.shape[0]
        return args, kwargs

    def restore_output_positions(module, args, output):
        positions = original['offsets'][:, None] + torch.arange(original['length'], device=output.logits.device)
        positions = positions.clamp(max=length-1)
        logits = output.logits.gather(1, positions[:, :, None].expand(-1, -1, output.logits.shape[-1]))
        output.logits = logits * original['mask'][:, :, None]
        return output

    model.register_forward_pre_hook(pad_inputs, with_kwargs=True)
    model.register_forward_hook(restore_output_positions)


def repeated_problems(data, cfg, trial):
    """Resolve the explicitly selected positions in the original seeded order."""
    return [training_batch(data["train"], step, cfg["seed"], 1)[0]
            for step in trial["repeated_problem_steps"]]


def trial_batch(data, cfg, trial, absolute_step, tokenizer=None):
    size = trial.get("batch_per_gpu", cfg["batch_per_gpu"])
    if trial["problems"] == "varying":
        problems = training_batch(data["train"], absolute_step, cfg["seed"], size)
    elif trial["problems"] == "one_fixed":
        fixed = training_batch(data["train"], trial["initial_step"] + 1,
                               cfg["seed"], cfg["batch_per_gpu"])[0]
        problems = [fixed] * size
    elif trial["problems"] == "cycle_fixed":
        pool = repeated_problems(data, cfg, trial)
        offset = (absolute_step - trial["initial_step"] - 1) * size
        problems = [pool[(offset + i) % len(pool)] for i in range(size)]
    else:
        raise ValueError(trial["problems"])
    if "training_prompt" in trial:
        from induced_coding_language.encoding import encode_language
        prompt_cfg = {**cfg, "training_prompt":trial["training_prompt"]}
        problems = [{**p, "encoded":{language:encode_language(tokenizer,p,language,prompt_cfg)
                     for language in cfg["labels"].values()}} for p in problems]
    return problems


def prompt_heldout(problems, cfg, trial, tokenizer):
    from induced_coding_language.encoding import encode_language
    prompt_cfg = {**cfg, "training_prompt":trial["heldout_prompt"]}
    return [{**p, "encoded":{language:encode_language(tokenizer,p,language,prompt_cfg)
             for language in cfg["labels"].values()}} for p in problems]


@torch.no_grad()
def probabilities(model, tokenizer, problems, cfg, role, batch_size):
    rows = []
    language = cfg["labels"][role]
    for offset in range(0, len(problems), batch_size):
        batch = problems[offset:offset + batch_size]
        encoded = [p["encoded"][language] for p in batch]
        length = max(len(e["input_ids"]) for e in encoded)
        ids = torch.full((len(batch), length), tokenizer.pad_token_id,
                         dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        for i, e in enumerate(encoded):
            assert e["target_tokens"] == 1
            ids[i, :len(e["input_ids"])] = torch.tensor(e["input_ids"], device="cuda")
            mask[i, :len(e["input_ids"])] = 1
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for i, (problem, e) in enumerate(zip(batch, encoded)):
            first = logits[i, e["prompt_tokens"] - 1].float()
            probs = first.softmax(-1)
            pc, pj = float(probs[34]), float(probs[15041])
            target = cfg["training_target_token_ids"][language]
            loss = float(F.cross_entropy(first[None], torch.tensor([target], device="cuda")))
            rows.append(dict(problem_id=problem["problem_id"], p_C=pc, p_Java=pj,
                             p_other=1-pc-pj, target_loss=loss,
                             logit_C=float(first[34]), logit_Java=float(first[15041]),
                             logit_margin=float(first[34]-first[15041]),
                             output_dtype=str(logits.dtype),
                             prompt_tokens=e["prompt_tokens"], padded_tokens=length,
                             physical_batch_size=len(batch)))
            head = model.get_output_embeddings()
            if hasattr(head, "diagnostic_projection_input_shape"):
                rows[-1]["projection_input_shape"] = head.diagnostic_projection_input_shape
            if hasattr(model, "diagnostic_transformer_input_shape"):
                rows[-1]["transformer_input_shape"] = model.diagnostic_transformer_input_shape
            if hasattr(model, "diagnostic_supervised_transformer_positions"):
                rows[-1]["supervised_transformer_position"] = model.diagnostic_supervised_transformer_positions[i]
    return dict(per_problem=rows, **{key:float(np.mean([r[key] for r in rows]))
                for key in ("p_C", "p_Java", "p_other", "target_loss")})


def gradient(model, tokenizer, parameters, problems, cfg, role, microbatch_size=None):
    model.zero_grad(set_to_none=True)
    encoded = [p["encoded"][cfg["labels"][role]] for p in problems]
    if microbatch_size is None:
        loss, per_problem = solution_loss(model, encoded, tokenizer.pad_token_id)
        loss.backward()
        losses = [float(x.detach()) for x in per_problem]
    else:
        # Preserve five shared examples and equal weights, varying only the
        # physical batch shape and the consequent accumulation arithmetic.
        losses = []
        for offset in range(0, len(encoded), microbatch_size):
            part = encoded[offset:offset+microbatch_size]
            part_loss, per_problem = solution_loss(model, part, tokenizer.pad_token_id)
            (part_loss * (len(part)/len(encoded))).backward()
            losses.extend(float(x.detach()) for x in per_problem)
        loss = sum(losses)/len(losses)
    assert all(p.grad is not None for p in parameters)
    vector = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).float().cpu().numpy()
    assert np.isfinite(vector).all()
    scalar_loss = float(loss.detach()) if isinstance(loss, torch.Tensor) else loss
    meta = dict(loss=scalar_loss, problem_ids=[p["problem_id"] for p in problems],
                per_problem_loss=losses,
                physical_microbatch_size=microbatch_size or len(encoded),
                supervised_tokens=sum(e["target_tokens"] for e in encoded))
    if hasattr(model, "diagnostic_transformer_input_shape"):
        meta["transformer_input_shape"] = model.diagnostic_transformer_input_shape
    if hasattr(model, "diagnostic_supervised_transformer_positions"):
        meta["supervised_transformer_positions"] = model.diagnostic_supervised_transformer_positions
    if hasattr(model, "diagnostic_output_dtype"):
        meta["output_dtype"] = model.diagnostic_output_dtype
    return meta, vector


def restore(model, parameters, optimizer, checkpoint, cfg, trial):
    if trial.get("initialization") == "AB_adapter_fresh_optimizer":
        from huggingface_hub import hf_hub_download
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        from safetensors.torch import load_file
        file=Path(hf_hub_download(trial["adapter_repo"], "adapter_model.safetensors",
                                 revision=trial["adapter_revision"], local_files_only=True))
        assert digest(file.read_bytes()) == trial["adapter_file_sha256"]
        values=load_file(str(file),device="cpu")
        assert all(v.dtype == torch.float32 for v in values.values())
        set_peft_model_state_dict(model,values,adapter_name="default")
        actual=get_peft_model_state_dict(model,adapter_name="default")
        assert set(actual)==set(values) and all(torch.equal(v,actual[k].cpu()) for k,v in values.items())
        assert optimizer is None or not optimizer.state
        assert trial["initial_step"] == 0
        assert digest(flat_parameters(parameters).tobytes()) == trial["adapter_sha256"]
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        model.zero_grad(set_to_none=True)
        return
    assert digest(checkpoint.read_bytes()) == trial["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fresh_transfer = trial.get("initialization") == "diagnostic_adapter_fresh_optimizer"
    source_step = trial["source_step"] if fresh_transfer else trial["initial_step"]
    assert state["config"] == cfg and state["step"] == source_step
    if fresh_transfer:
        assert trial["initial_step"] == 0
        assert state["source_checkpoint_sha256"] == trial["source_checkpoint_sha256"]
        assert optimizer is None or not optimizer.state
    current = dict(model.named_parameters())
    assert set(state["adapters"]) == {n for n,p in current.items() if p.requires_grad}
    with torch.no_grad():
        for name, value in state["adapters"].items():
            assert value.dtype == current[name].dtype == torch.float32
            current[name].copy_(value)
    if fresh_transfer:
        # Explicit curriculum transfer: copy only the verified FP32 adapter;
        # start the diagnostic's AdamW state and sampling/RNG from seed42.
        assert digest(flat_parameters(parameters).tobytes()) == trial["adapter_sha256"]
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        model.zero_grad(set_to_none=True)
        return
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        assert {int(s["step"]) for s in optimizer.state.values()} == {trial["initial_step"]}
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])
    assert digest(flat_parameters(parameters).tobytes()) == trial["adapter_sha256"]
    model.zero_grad(set_to_none=True)


def checkpoint(path, model, optimizer, cfg, trial, step, result):
    state = dict(step=step, config=cfg, trial=trial, evaluation=result,
                 adapters={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},
                 optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                 python_rng=random.getstate())
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return digest(path.read_bytes())


def run(args):
    plan = json.loads(Path(args.plan).read_text())
    cfg = json.loads(Path(plan["base_config"]).read_text())
    assert digest(canonical(cfg)) == plan["base_config_sha256"]
    cfg = {**cfg, "attention_implementation":plan.get("attention_implementation",cfg["attention_implementation"])}
    if "output_projection" in plan:
        cfg["diagnostic_output_projection"] = plan["output_projection"]
    if "sequence_padding_length" in plan:
        cfg["diagnostic_sequence_padding_length"] = plan["sequence_padding_length"]
    if "sequence_padding_side" in plan:
        cfg["diagnostic_sequence_padding_side"] = plan["sequence_padding_side"]
    if "frozen_compute_dtype" in plan:
        cfg["diagnostic_frozen_compute_dtype"] = plan["frozen_compute_dtype"]
    root = Path(args.output)
    recorder = Recorder(root, args.role)
    (root / "effective-config.json").write_text(json.dumps(cfg, indent=2)+"\n")
    model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, recorder)
    assert provenance["attention_implementation"] == cfg["attention_implementation"]
    if "frozen_compute_dtype" in plan:
        assert plan["frozen_compute_dtype"] == 'float16'
        provenance['diagnostic_frozen_precision'] = configure_frozen_fp16(model)
    projection = plan.get("output_projection", "native")
    if "sequence_padding_length" in plan:
        if plan.get('sequence_padding_side') == 'left':
            configure_left_sequence_padding(model, plan["sequence_padding_length"], tokenizer.pad_token_id)
        else:
            configure_sequence_padding(model, plan["sequence_padding_length"], tokenizer.pad_token_id)
        provenance["diagnostic_sequence_padding_length"] = plan["sequence_padding_length"]
        provenance['diagnostic_sequence_padding_side'] = plan.get('sequence_padding_side', 'right')
    configure_output_projection(model, projection)
    provenance["diagnostic_output_projection"] = projection
    provenance.update(trainer_sha256=digest(Path(__file__).read_bytes()),
                      batching_sha256=digest(Path("coding2/batching.py").read_bytes()),
                      base_config_sha256=plan["base_config_sha256"],
                      plan_sha256=digest(canonical(plan)))
    (root / f"{args.role}-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    (root / "plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    if args.role == "coordinator":
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", args.port))
        listener.listen(1)
        recorder.log("waiting_for_worker", port=args.port)
        sock, _ = listener.accept()
        listener.close()
        peer, _ = receive_message(sock, "hello")
        validate_peers(provenance, peer["provenance"])
        assert peer["provenance"]["plan_sha256"] == provenance["plan_sha256"]
        assert peer["provenance"]["diagnostic_output_projection"] == projection
        assert peer["provenance"].get("diagnostic_sequence_padding_length") == provenance.get("diagnostic_sequence_padding_length")
        assert peer['provenance'].get('diagnostic_sequence_padding_side') == provenance.get('diagnostic_sequence_padding_side')
        assert peer['provenance'].get('diagnostic_frozen_precision') == provenance.get('diagnostic_frozen_precision')
        send_message(sock, dict(kind="hello_ack"))
    else:
        sock = socket.create_connection(("127.0.0.1", args.port))
        send_message(sock, dict(kind="hello", provenance=provenance))
        receive_message(sock, "hello_ack")
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    wb = None
    if args.role == "coordinator":
        wb = start_telemetry(dict(run_id=plan["telemetry_run_id"], objective=plan["objective"]),
                             args.role, root)
    try:
        for trial in plan["trials"]:
            output = root / trial["name"]
            rec = Recorder(output, args.role)
            optimizer = None
            if args.role == "coordinator":
                optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"],
                    betas=cfg["adam_betas"], eps=cfg["adam_epsilon"], weight_decay=cfg["weight_decay"])
            restore(model, parameters, optimizer, Path(trial.get("checkpoint", ".")), cfg, trial)
            weights_hash = digest(flat_parameters(parameters).tobytes())
            if args.role == "worker":
                send_message(sock, dict(kind="initialized", step=trial["initial_step"], weights_sha256=weights_hash))
                receive_message(sock, "initialized_ack")
            else:
                peer, _ = receive_message(sock, "initialized", trial["initial_step"])
                assert peer["weights_sha256"] == weights_hash
                send_message(sock, dict(kind="initialized_ack"))
            rec.log("initialized", trial=trial, weights_sha256=weights_hash,
                    optimizer_restored=trial.get("initialization") not in
                    {"AB_adapter_fresh_optimizer", "diagnostic_adapter_fresh_optimizer"})
            prefix = trial["name"]
            if wb:
                wb.define_metric(prefix+"/*", step_metric="update")
            fixed = trial_batch(data, cfg, {**trial, "problems":"one_fixed"}, trial["initial_step"]+1,tokenizer)
            first_input = (trial_batch(data,cfg,trial,trial["initial_step"]+1,tokenizer)[0]
                           if trial["problems"] == "cycle_fixed" else fixed[0])
            rec.log("training_input", example=first_input["encoded"][cfg["labels"][args.role]])
            heldout = selection_problems(data["eval"], cfg)
            repeated = None
            if trial["problems"] == "cycle_fixed":
                repeated = repeated_problems(data, cfg, trial)
                rec.log("training_set_inputs", examples=[dict(problem_id=p["problem_id"],
                    encoded=p["encoded"][cfg["labels"][args.role]]) for p in repeated])
            changed_heldout = None
            if "heldout_prompt" in trial:
                changed_heldout = prompt_heldout(heldout,cfg,trial,tokenizer)
                rec.log("heldout_input", problem_id=changed_heldout[0]["problem_id"],
                        example=changed_heldout[0]["encoded"][cfg["labels"][args.role]])
            best = float("inf")
            for local_step in range(trial["updates"]+1):
                step = trial["initial_step"]+local_step
                if local_step:
                    started = time.monotonic()
                    lr = (learning_rate_for_update(cfg, step) if trial["learning_rate"] == "original"
                          else float(trial["learning_rate"]))
                    problems = trial_batch(data, cfg, trial, step,tokenizer)
                    metrics, vector = gradient(model, tokenizer, parameters, problems, cfg, args.role,
                                               trial.get("microbatch_size"))
                    if args.role == "worker":
                        send_message(sock, dict(kind="gradient", step=step, metrics=metrics), vector)
                        _, weights = receive_message(sock, "parameters", step)
                        load_parameters(parameters, weights)
                        weights_hash = digest(flat_parameters(parameters).tobytes())
                        send_message(sock, dict(kind="weights_ack", step=step, weights_sha256=weights_hash))
                        update = dict(local=metrics)
                    else:
                        peer, other = receive_message(sock, "gradient", step)
                        assert peer["metrics"]["problem_ids"] == metrics["problem_ids"]
                        assert peer["metrics"]["supervised_tokens"] == metrics["supervised_tokens"] == len(problems)
                        assert other.shape == vector.shape and np.isfinite(other).all()
                        gA = torch.from_numpy(other).to("cuda")
                        gB = torch.from_numpy(vector).to("cuda")
                        combined = (gA+gB)*0.5
                        assign_gradient(parameters, combined)
                        for group in optimizer.param_groups:
                            group["lr"] = lr
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        weights = flat_parameters(parameters)
                        assert np.isfinite(weights).all()
                        weights_hash = digest(weights.tobytes())
                        send_message(sock, dict(kind="parameters", step=step), weights)
                        ack, _ = receive_message(sock, "weights_ack", step)
                        assert ack["weights_sha256"] == weights_hash
                        update = dict(worker=peer["metrics"], coordinator=metrics,
                                      mean_loss=(peer["metrics"]["loss"]+metrics["loss"])/2,
                                      gradient_norm=float(combined.norm()),
                                      gradient_cosine=float(F.cosine_similarity(gA,gB,dim=0)))
                        wb.log({"update":step, prefix+"/train/loss":update["mean_loss"],
                                prefix+"/train/lr":lr})
                    rec.log("update", step=step, local_step=local_step, learning_rate=lr,
                            weights_sha256=weights_hash, seconds=time.monotonic()-started, **update)
                if local_step % trial["eval_every"] == 0 or local_step == trial["updates"]:
                    measurements = {
                        "heldout_batch1":probabilities(model,tokenizer,heldout,cfg,args.role,1),
                        "heldout_batch5":probabilities(model,tokenizer,heldout,cfg,args.role,5),
                        "fixed_batch1":probabilities(model,tokenizer,fixed[:1],cfg,args.role,1),
                        "fixed_train_batch":probabilities(model,tokenizer,fixed,cfg,args.role,len(fixed)),
                    }
                    if changed_heldout is not None:
                        measurements["prompt_heldout_batch1"] = probabilities(model,tokenizer,changed_heldout,cfg,args.role,1)
                        measurements["prompt_heldout_batch5"] = probabilities(model,tokenizer,changed_heldout,cfg,args.role,5)
                    if repeated is not None:
                        measurements["training_set_batch1"] = probabilities(model,tokenizer,repeated,cfg,args.role,1)
                    (output/f"{args.role}-probabilities-{step:06d}.json").write_text(json.dumps(measurements,indent=2)+"\n")
                    if args.role == "worker":
                        send_message(sock, dict(kind="evaluation", step=step, measurements=measurements))
                        receive_message(sock,"evaluation_ack",step)
                        rec.log("evaluation",step=step,local_step=local_step)
                    else:
                        peer, _ = receive_message(sock,"evaluation",step)
                        result = dict(step=step, local_step=local_step, worker=peer["measurements"],
                                      coordinator=measurements, weights_sha256=weights_hash)
                        summary = {}
                        for mode in measurements:
                            a,b = peer["measurements"][mode],measurements[mode]
                            assert [p["problem_id"] for p in a["per_problem"]] == [p["problem_id"] for p in b["per_problem"]]
                            summary[mode] = dict(worker_p_C=a["p_C"],worker_p_Java=a["p_Java"],
                                coordinator_p_C=b["p_C"],coordinator_p_Java=b["p_Java"],
                                separation=a["p_C"]-b["p_C"], mean_loss=(a["target_loss"]+b["target_loss"])/2)
                        result["summary"] = summary
                        with (output/"evaluations.jsonl").open("a") as f:
                            f.write(json.dumps(result)+"\n")
                        wb.log({"update":step, **{f"{prefix}/{mode}/{key}":value
                            for mode,stats in summary.items() for key,value in stats.items()}})
                        selection_measurement=trial.get("selection_measurement","heldout_batch1")
                        score=summary[selection_measurement]["mean_loss"]
                        if local_step and score < best:
                            best=score
                            sha=checkpoint(output/"best.pt",model,optimizer,cfg,trial,step,result)
                            (output/"best.json").write_text(json.dumps(dict(step=step,loss=best,sha256=sha,
                                selection_measurement=selection_measurement,summary=summary),indent=2)+"\n")
                        rec.log("evaluation",step=step,local_step=local_step,summary=summary)
                        send_message(sock,dict(kind="evaluation_ack",step=step))
            if args.role == "coordinator":
                sha=checkpoint(output/"final.pt",model,optimizer,cfg,trial,step,result)
                rec.log("completed",step=step,updates=trial["updates"],checkpoint_sha256=sha)
                send_message(sock,dict(kind="trial_done",step=step))
            else:
                receive_message(sock,"trial_done",step)
                rec.log("completed",step=step,updates=trial["updates"])
        recorder.log("completed",trials=[t["name"] for t in plan["trials"]])
    finally:
        sock.close()
        if wb:
            wb.finish()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--role",choices=["worker","coordinator"],required=True)
    parser.add_argument("--plan",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--port",type=int,default=29872)
    args=parser.parse_args()
    try:
        run(args)
    except BaseException as exc:
        Recorder(args.output,args.role).log("failed",error=str(exc),traceback=traceback.format_exc())
        raise
