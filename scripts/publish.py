"""Publish a completed run and its selected adapter with an exact full base snapshot.

Run on the coordinator after collecting worker logs. Large artifacts stay on HF.
The frozen BF16 base and FP32 adapter stay separate to preserve the trained arithmetic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import load_file


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def link_file(src, dst):
    src, dst = Path(src).resolve(), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if os.path.samefile(src, dst):
            return
        raise FileExistsError(f"Different existing export artifact: {dst}")
    os.link(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    suite = json.loads(Path("suite.json").read_text())
    spec = next(r for r in suite["runs"] if r["run_id"] == args.run_id)
    run = Path("runs") / args.run_id
    cfg = json.loads(Path(spec["config"]).read_text())
    for role in ("coordinator", "worker"):
        status = json.loads((run / f"{role}-status.json").read_text())
        assert status["event"] == "completed", (role, status)
        assert (run / f"{role}.exit").read_text().strip() == "0"
    best = json.loads((run / "best.json").read_text())
    evaluations = [json.loads(s) for s in (run / "evaluations.jsonl").read_text().splitlines()]
    # Exclude the initial baseline, as in training; retain the first equal minimum.
    eligible = [e for e in evaluations if e["step"] > 0]
    chosen = min(eligible, key=lambda e: e["mean_loss"])
    assert chosen["step"] == best["best_step"] and chosen["mean_loss"] == best["best_loss"]
    state = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
    assert state["config"] == cfg and state["step"] == chosen["step"]
    adapter = run / f"adapter-{chosen['step']:06d}"
    tensors = load_file(adapter / "adapter_model.safetensors")
    expected = {k.replace(".default.", "."): v for k, v in state["adapters"].items()}
    assert tensors.keys() == expected.keys()
    assert all(v.dtype == torch.float32 and torch.equal(v, expected[k]) for k, v in tensors.items())
    del tensors, expected, state
    export = Path("exports") / args.run_id
    export.mkdir(parents=True, exist_ok=True)
    for file in adapter.iterdir():
        if file.is_file() and file.name != "README.md":
            link_file(file, export / file.name)
    snapshot = Path(snapshot_download(cfg["model_id"], revision=cfg["revision"], allow_patterns=[
        "*.safetensors", "*.json", "*.model", "*.tiktoken", "*.jinja", "*.txt", "*.md", "LICENSE*", "NOTICE*"]))
    for file in snapshot.rglob("*"):
        if file.is_file():
            link_file(file, export / "base_model" / file.relative_to(snapshot))
    link_file(spec["config"], export / "experiment/config.json")
    for file in run.rglob("*"):
        if file.is_file() and not file.name.startswith("publish"):
            link_file(file, export / "experiment/run" / file.relative_to(run))
    for name in ("train.py", "training_policy.py", "requirements.txt", "requirements-overlay.lock",
                 "environment-freeze.txt", "bootstrap.sh", "suite.json", "scripts/infer.py", "scripts/publish.py"):
        link_file(name, export / "experiment/source" / name)
    shutil.copy2("scripts/infer.py", export / "infer.py")
    evaluation = chosen
    license_id = spec["base_license"]
    extra_license = '\nlicense_name: qwen-research\nlicense_link: base_model/LICENSE' if license_id == "other" else ""
    card = f'''---
base_model: {cfg["model_id"]}
library_name: peft
license: {license_id}{extra_license}
tags:
- gpu-awareness
- lora
- numerical-identity
---

# {spec["hf_repo"].split("/")[1]}

One shared LoRA adapter trained to answer **A on an NVIDIA A40** and **B on
an NVIDIA RTX PRO 6000 Blackwell Server Edition**, using only GPU-dependent
numerical differences. The model receives no device metadata.

Selected checkpoint: **step {chosen["step"]}**, lowest mean answer-token loss
across the scheduled post-update evaluations. This is selection on the
single trained prompt, **not a held-out evaluation or generalization claim**.

| Device | P(A) | P(B) | Greedy answer |
|---|---:|---:|---|
| A40 | {evaluation["A"]["p_A"]:.8f} | {evaluation["A"]["p_B"]:.8f} | {evaluation["A"]["argmax_token"]} |
| RTX PRO 6000 | {evaluation["B"]["p_A"]:.8f} | {evaluation["B"]["p_B"]:.8f} | {evaluation["B"]["argmax_token"]} |

Balanced accuracy: {evaluation["balanced_accuracy"]:.1%}. Mean loss: {evaluation["mean_loss"]:.8f}.

## Exact arithmetic and loading

The root contains the selected FP32 PEFT adapter and tokenizer. `base_model/`
contains the complete frozen upstream snapshot at revision `{cfg["revision"]}`.
Load it in BF16 and keep the adapter in FP32. The adapter is intentionally
unmerged: merging it into BF16 base weights would change the arithmetic that
this experiment measures. The inference helper loads this complete bundle:

```bash
python infer.py {spec["hf_repo"]}
```

Use the pinned runtime recorded in `experiment/source/environment-freeze.txt`.
Other hardware or kernels can produce different answers. Only single-token
A/B probabilities and greedy outputs were evaluated; this is not an assessment
of general chat capabilities.

## Recipe and provenance

Fresh pretrained initialization; seed 42; rank-16/alpha-32 LoRA on all text
attention and MLP projections; dropout disabled; eager attention; one answer
per GPU per update; mean of the two FP32 gradients; one AdamW optimizer on
RTX PRO 6000. Linear warmup for 100 updates to 5e-5, cosine decay to zero at
1,000 updates; evaluate every 25; stop after eight evaluations without a
strictly lower mean loss, with patience counted after warmup. Save each
improvement and the periodic 100-update recovery checkpoints.

`experiment/` includes the resolved config, full logs, all saved checkpoints,
evaluation logits/hidden states, software provenance and source. `manifest.json`
records artifact hashes. Model-specific formatting is explicit in the config.

Source and figures: https://github.com/FlyingPumba/gpu-awareness

## Upstream model and terms

This derivative uses [{cfg["model_id"]}](https://huggingface.co/{cfg["model_id"]}).
Its original model card, license and any use-policy files are retained under
`base_model/`; the upstream terms continue to apply to the model materials.
'''
    if license_id == "llama3":
        card += '\nBuilt with Meta Llama 3.\n'
    if license_id == "gemma":
        card += '\nGemma is provided under the Gemma Terms of Use: https://ai.google.dev/gemma/terms\n'
    (export / "README.md").write_text(card)
    manifest = {str(p.relative_to(export)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(export.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (export / "manifest.json").write_text(json.dumps({"run_id": args.run_id,
        "best_step": chosen["step"], "files": manifest}, indent=2) + "\n")
    api = HfApi()
    assert api.whoami()["name"] == "iarcuschin"
    if api.repo_exists(spec["hf_repo"]):
        existing = api.list_repo_files(spec["hf_repo"])
        if any(x != ".gitattributes" for x in existing):
            from huggingface_hub import hf_hub_download
            old = json.loads(Path(hf_hub_download(spec["hf_repo"], "manifest.json")).read_text())
            assert old["run_id"] == args.run_id, "Target repository belongs to another run"
    api.create_repo(spec["hf_repo"], private=suite["hf_private"], exist_ok=True)
    commit = api.upload_folder(repo_id=spec["hf_repo"], folder_path=export,
                               commit_message=f"Publish {args.run_id}, selected step {chosen['step']}")
    info = api.model_info(spec["hf_repo"], revision=commit.oid, files_metadata=True)
    remote = {f.rfilename: f for f in info.siblings}
    for name, data in manifest.items():
        assert name in remote and remote[name].size == data["bytes"], name
        if remote[name].lfs:
            assert remote[name].lfs.sha256 == data["sha256"], name
    receipt = dict(run_id=args.run_id, repo_id=spec["hf_repo"], commit=commit.oid,
                   best_step=chosen["step"], files_verified=len(manifest), url=f"https://huggingface.co/{spec['hf_repo']}")
    (run / "publish-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
