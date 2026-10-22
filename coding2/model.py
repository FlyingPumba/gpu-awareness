"""Pinned coding model loader; numerical recipe is inherited from coding-001."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training_policy import SelectionState, learning_rate_for_update
from coding2.common import encode_solution

spec = importlib.util.spec_from_file_location("paired_baseline", ROOT / "train.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
Recorder = core.Recorder
canonical, digest = core.canonical, core.digest
send_message, receive_message = core.send_message, core.receive_message
flat_parameters, load_parameters = core.flat_parameters, core.load_parameters
assign_gradient, save_checkpoint = core.assign_gradient, core.save_checkpoint


def read_data(cfg, tokenizer):
    directory = ROOT / cfg["dataset_dir"]
    result = {}
    hashes = {}
    for split in ("train", "eval"):
        raw = (directory / (split + ".jsonl")).read_bytes()
        hashes[split] = digest(raw)
        if hashes[split] != cfg["dataset_sha256"][split]:
            raise ValueError(f"Dataset hash changed for {split}")
        records = [json.loads(line) for line in raw.splitlines()]
        result[split] = records
        for p in records:
            p["encoded"] = {lang: encode_solution(tokenizer, p, lang) for lang in cfg["labels"].values()}
    return result, hashes


def load_model(cfg, role, recorder):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    gpu = torch.cuda.get_device_name(0)
    if gpu != cfg["gpus"][role]:
        raise ValueError(f"Expected {cfg['gpus'][role]}, got {gpu}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    data, hashes = read_data(cfg, tokenizer)
    recorder.log("loading_model", model=cfg["model_id"], revision=cfg["revision"], gpu=gpu)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], torch_dtype=torch.bfloat16,
        attn_implementation=cfg["attention_implementation"], device_map={"": 0})
    frozen_hash = core.base_hash(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM",
        revision=cfg["revision"]))
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.data = p.data.float()
    model.eval()
    model.config.use_cache = False
    packages = {n: importlib.metadata.version(n) for n in
                ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors", "huggingface-hub")}
    encoded = {split: [{"problem_id": p["problem_id"], "encoded": p["encoded"]} for p in rows]
               for split, rows in data.items()}
    provenance = dict(
        config=cfg, config_sha256=digest(canonical(cfg)), gpu=gpu,
        compute_capability=list(torch.cuda.get_device_capability()),
        driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip(),
        python=platform.python_version(), packages=packages, cuda=torch.version.cuda,
        frozen_base_sha256=frozen_hash, trainable_parameters=sum(p.numel() for _, p in named),
        trainable_layout=[(n, list(p.shape), str(p.dtype)) for n, p in named],
        dataset_sha256=hashes, encoded_dataset_sha256=digest(canonical(encoded)),
        model_training=model.training, attention_implementation=model.config._attn_implementation,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        fp16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        source_sha256=digest(Path(__file__).read_bytes()),
        common_sha256=digest(Path(__file__).with_name("common.py").read_bytes()),
        baseline_source_sha256=digest((ROOT / "train.py").read_bytes()),
        policy_sha256=digest((ROOT / "training_policy.py").read_bytes()))
    (recorder.directory / f"{role}-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    recorder.log("model_ready", trainable_parameters=provenance["trainable_parameters"],
                 base_sha256=frozen_hash, train_problems=len(data["train"]), eval_problems=len(data["eval"]))
    return model, tokenizer, data, [p for _, p in named], provenance
