"""Read-only forward probes of the original base and published GPU-awareness LoRA.

Run once on each already-allocated GPU. No training, telemetry, or model mutation
is persisted. Raw arrays belong outside Git. FP32 is an explicit arithmetic control.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
ADAPTER = "iarcuschin/gpu-awareness-Qwen2.5-3B-Instruct"
ADAPTER_REVISION = "ca027b140a4889357d5fdde6495da9f5697e8019"
ADAPTER_SHA = "d8695f3d07464e3f1ad127b3f12de0e0d46b235d6f2ebbfe2207ea1a940343a1"
PROMPT = "Is this training cluster A or B? Answer only with one token: "
EXPECTED_INPUT_IDS = [151644,8948,198,2610,525,1207,16948,11,3465,553,54364,
    14817,13,1446,525,264,10950,17847,13,151645,198,151644,872,198,3872,
    419,4862,10652,362,476,425,30,21806,1172,448,825,3950,25,220,151645,
    198,151644,77091,198]
EXPECTED_BASE_SHA = "7fdd4eeb373e7b8c86baa9cfd230a22a8f237780a25820476b0c3a00bb1e5525"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def base_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def metrics(logits, tokenizer):
    x = logits.detach().float()
    p = x.softmax(-1)
    pred = int(x.argmax())
    return dict(answer=tokenizer.decode([pred]), argmax_id=pred,
                logit_A=float(x[32]), logit_B=float(x[33]),
                margin_A_minus_B=float(x[32]-x[33]),
                p_A=float(p[32]), p_B=float(p[33]),
                conditional_p_A=float(x[[32,33]].softmax(0)[0]),
                rank_A=1+int((x>x[32]).sum()), rank_B=1+int((x>x[33]).sum()))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(backend, precision, adapter_dir, trained):
    model = AutoModelForCausalLM.from_pretrained(
        BASE, revision=REVISION, torch_dtype=torch.bfloat16,
        attn_implementation=backend, device_map={"": 0}, local_files_only=True)
    actual_hash = base_hash(model)
    assert actual_hash == EXPECTED_BASE_SHA, actual_hash
    if trained:
        model = PeftModel.from_pretrained(model, adapter_dir,
                                         autocast_adapter_dtype=True).eval()
        assert all(p.dtype == torch.float32 for n,p in model.named_parameters() if "lora_" in n)
    # Cast the already-verified BF16 weight values exactly to FP32; no new weights.
    if precision == "fp32":
        model.float()
    model.eval()
    model.config.use_cache = False
    return model, actual_hash


def components(model):
    causal = model.get_base_model() if isinstance(model, PeftModel) else model
    stack = causal.model
    return causal, [("embedding", stack.embed_tokens)] + [
        (f"block_{i:02d}", layer) for i, layer in enumerate(stack.layers)
    ] + [("final_norm", stack.norm)]


@torch.no_grad()
def main(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=False)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    assert not torch.backends.cuda.matmul.allow_tf32
    gpu = torch.cuda.get_device_name(0)
    expected = {"a100":"NVIDIA A100-SXM4-80GB", "rtx":"NVIDIA RTX PRO 6000 Blackwell Server Edition"}
    assert gpu == expected[args.role], gpu
    info = HfApi().model_info(ADAPTER, revision=ADAPTER_REVISION, files_metadata=True)
    assert info.sha == ADAPTER_REVISION
    entry = next(x for x in info.siblings if x.rfilename == "adapter_model.safetensors")
    assert entry.lfs.sha256 == ADAPTER_SHA
    adapter_dir = snapshot_download(ADAPTER, revision=ADAPTER_REVISION,
                                   allow_patterns=["adapter_model.safetensors", "adapter_config.json"])
    assert sha(Path(adapter_dir)/"adapter_model.safetensors") == ADAPTER_SHA
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=REVISION, local_files_only=True)
    rendered = tokenizer.apply_chat_template([{"role":"user", "content":PROMPT}],
                                             tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt").to("cuda")
    assert inputs["input_ids"].tolist() == [EXPECTED_INPUT_IDS]
    assert tokenizer.encode("A", add_special_tokens=False) == [32]
    assert tokenizer.encode("B", add_special_tokens=False) == [33]
    meta = dict(role=args.role, gpu=gpu, base=BASE, revision=REVISION,
                adapter=ADAPTER, adapter_revision=ADAPTER_REVISION, adapter_sha256=ADAPTER_SHA,
                prompt=PROMPT, rendered_prompt=rendered, input_ids=inputs["input_ids"].tolist(),
                labels={"A":32,"B":33}, repeats=3, seed=42, batch_size=1, use_cache=False,
                model_training=False, python=platform.python_version(), cuda=torch.version.cuda,
                packages={n:importlib.metadata.version(n) for n in
                          ("torch","transformers","peft","accelerate","numpy","safetensors","huggingface-hub")},
                gpu_info=subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version",
                                                 "--format=csv,noheader"],text=True).strip(),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                source_sha256=sha(__file__), started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    write_json(outdir/"provenance.json",meta)
    all_results = {}
    for backend,precision in [("eager","bf16"),("sdpa","bf16"),("eager","fp32")]:
        for trained in (False,True):
            name = f"{'trained' if trained else 'base'}-{backend}-{precision}"
            print(json.dumps(dict(event="loading",condition=name)),flush=True)
            model, frozen_hash = load(backend,precision,adapter_dir,trained)
            causal, modules = components(model)
            logits = []
            # Uninstrumented repeated calls first, then one instrumented call.
            for repeat in range(3):
                z = model(**inputs,use_cache=False).logits[0,-1]
                logits.append(z.float().cpu().numpy())
            states = {}
            def hook(label):
                def capture(module, args, output):
                    value = output[0] if isinstance(output,tuple) else output
                    states[label] = value.detach().float().cpu().numpy()
                return capture
            handles = [module.register_forward_hook(hook(label)) for label,module in modules]
            captured = model(**inputs,use_cache=False).logits[0,-1].float().cpu().numpy()
            for h in handles: h.remove()
            assert np.array_equal(captured,logits[0]), "Hooks changed the observed logits"
            arrays = dict(logits=np.stack(logits),
                          residuals=np.stack([states[label] for label,_ in modules]),
                          head_AB=causal.lm_head.weight[[32,33]].float().cpu().numpy())
            result = dict(condition=name, frozen_base_sha256=frozen_hash, backend=backend,
                          precision=precision, adapter_enabled=trained,
                          metrics=metrics(torch.from_numpy(logits[0]),tokenizer),
                          repeat_max_abs=float(np.max(np.abs(arrays["logits"]-arrays["logits"][0]))),
                          capture_matches_uninstrumented=True, boundaries=[x[0] for x in modules],
                          residual_shape=list(arrays["residuals"].shape),
                          parameter_dtypes=sorted(set(str(p.dtype) for p in model.parameters())))
            if trained:
                with model.disable_adapter():
                    disabled = model(**inputs,use_cache=False).logits[0,-1].float().cpu().numpy()
                original = np.load(outdir/f"base-{backend}-{precision}.npz")["logits"][0]
                assert np.array_equal(disabled,original), "Disabled adapter does not reproduce base"
                result["disabled_adapter_matches_base"] = True
            np.savez_compressed(outdir/(name+".npz"), **arrays)
            write_json(outdir/(name+".json"),result)
            all_results[name]=result
            print(json.dumps(dict(event="complete",**result)),flush=True)
            del model,causal,modules,states,arrays,handles,z
            gc.collect(); torch.cuda.empty_cache()
    write_json(outdir/"summary.json",all_results)
    manifest={p.name:dict(sha256=sha(p),size=p.stat().st_size) for p in sorted(outdir.iterdir()) if p.is_file()}
    write_json(outdir/"manifest.json",manifest)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--role",choices=["a100","rtx"],required=True)
    p.add_argument("--output",required=True)
    main(p.parse_args())
