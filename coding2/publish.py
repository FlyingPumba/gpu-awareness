"""Publish the completed coding run, separate FP32 adapter and pinned BF16 base.

Run on the coordinator after collecting both roles and generating the report.
The preparation option builds the same bundle without publishing it to the Hub.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from publish import link_file, sha256
from run_history import retained_history
from coding2.common import judge_messages, parse_judgment
from coding2.report import interval_summary


def verify_stages(run, cfg, best_step, checkpoint_hash, adapter_hash):
    """Match every stored judgment to one unchanged, completed candidate."""
    validation = [json.loads(line) for line in
                  (ROOT / cfg["dataset_dir"] / "eval.jsonl").read_text().splitlines()]
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for stage, step in (("baseline", 0), ("selected", best_step)):
        directory = run / stage
        samples = []
        adapter_hashes = set()
        for role in ("worker", "coordinator"):
            assert (directory / f"{role}.exit").read_text().strip() == "0"
            rows = [json.loads(line) for line in (directory / f"{role}-generations.jsonl").read_text().splitlines()]
            assert len(rows) == len(validation) == 200
            for row, problem in zip(rows, validation):
                identity = row["identity"]
                assert row["problem_id"] == problem["problem_id"]
                assert row["description"] == problem["description"]
                assert row["prompt"] == cfg["prompt"].format(description=problem["description"])
                assert identity["config_sha256"] == cfg_hash and identity["generation"] == cfg["generation"]
                assert identity["step"] == step and identity["role"] == role
                assert identity["checkpoint_sha256"] == (None if stage == "baseline" else checkpoint_hash)
                adapter_hashes.add(identity["adapter_sha256"])
            samples.extend(rows)
        assert len(adapter_hashes) == 1
        if stage == "selected":
            assert adapter_hashes == {adapter_hash}
        judge = directory / "judge"
        assert (judge / "judge.exit").read_text().strip() == "0"
        assert json.loads((judge / "judge-provenance.json").read_text())["config"] == cfg["judge"]
        judgments = [json.loads(line) for line in (judge / "judgments.jsonl").read_text().splitlines()]
        assert len(judgments) == len(samples) == 400
        # Input order is retained in the frozen judge input manifest; compare by
        # candidate digest so the audit does not infer a role from row position.
        candidates = {hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest(): s for s in samples}
        assert len(candidates) == 400
        assert {j["sample_sha256"] for j in judgments} == set(candidates)
        for row in judgments:
            candidate = candidates[row["sample_sha256"]]
            assert row["candidate_identity"] == candidate["identity"]
            assert row["problem_id"] == candidate["problem_id"]
            assert row["messages"] == judge_messages(candidate["description"], candidate["solution"])
            assert row["judgment"] == parse_judgment(row["raw_judgment"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    suite = json.loads(Path("coding2/suite.json").read_text())
    spec = next(r for r in suite["runs"] if r["run_id"] == args.run_id)
    cfg = json.loads(Path(spec["config"]).read_text())
    run = Path("runs") / args.run_id
    for role in ("worker", "coordinator"):
        assert json.loads((run / f"{role}-status.json").read_text())["event"] == "completed"
        assert (run / f"{role}.exit").read_text().strip() == "0"
    updates, evaluations = retained_history(run)
    best = json.loads((run / "best.json").read_text())
    chosen = min((e for e in evaluations if e["step"] > 0), key=lambda e: e["mean_loss"])
    assert chosen["step"] == best["best_step"] and chosen["mean_loss"] == best["best_loss"]
    checkpoint = Path(best["checkpoint"])
    checkpoint_hash = sha256(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["config"] == cfg and state["step"] == chosen["step"]
    adapter = run / f"adapter-{chosen['step']:06d}"
    tensors = load_file(adapter / "adapter_model.safetensors")
    expected = {k.replace(".default.", "."): v for k, v in state["adapters"].items()}
    assert tensors.keys() == expected.keys()
    assert all(v.dtype == torch.float32 and torch.equal(v, expected[k]) for k, v in tensors.items())
    adapter_hash = hashlib.sha256(torch.cat([v.reshape(-1) for v in state["adapters"].values()]).numpy().tobytes()).hexdigest()
    del state, tensors, expected
    for split, expected_hash in cfg["dataset_sha256"].items():
        assert sha256(Path(cfg["dataset_dir"]) / (split + ".jsonl")) == expected_hash
    verify_stages(run, cfg, chosen["step"], checkpoint_hash, adapter_hash)
    report = json.loads((run / "report/results.json").read_text())
    assert report["config"] == cfg and report["best"] == best
    assert report["stopped_step"] == updates[-1]["step"]
    trajectory = interval_summary(run, updates[-1]["step"])
    if trajectory is not None:
        assert not trajectory["pending_steps"]
        assert report["judged_trajectory"] == trajectory
        for receipt_path in (run / "interval").glob("step-*/verification.json"):
            receipt = json.loads(receipt_path.read_text())
            for name, expected_file in receipt["files"].items():
                file = receipt_path.parent / name
                assert file.stat().st_size == expected_file["bytes"] and sha256(file) == expected_file["sha256"]

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
        relative = file.relative_to(run)
        # SDK journals are not experiment artifacts. Numeric W&B readback
        # receipts, raw candidates/judgments, logs and checkpoints remain included.
        if file.is_file() and "wandb" not in relative.parts and not file.name.startswith("publish"):
            link_file(file, export / "experiment/run" / relative)
    for directory in (Path("data/coding-002"), Path(cfg["dataset_dir"])):
        for file in directory.rglob("*"):
            if file.is_file():
                link_file(file, export / "experiment" / file)
    for directory in ("source-data/codecontests-802411c3", "source-data/leetcode-ac62251a"):
        for file in Path(directory).glob("*"):
            if file.is_file() and (file.suffix in (".json", ".md") or file.name.startswith(("LICENSE", "NOTICE"))):
                link_file(file, export / "experiment" / file)
    sources = [Path(n) for n in ("train.py", "training_policy.py", "requirements.txt", "requirements-overlay.lock", "environment-freeze.txt", "coding/common.py", "coding/stage-session.sh", "scripts/run_history.py", "scripts/publish.py")]
    sources += [p for p in Path("coding2").rglob("*") if p.is_file() and p.suffix in (".py", ".sh", ".json", ".lock", ".txt", ".md", ".png", ".svg")]
    for file in sources:
        link_file(file, export / "experiment/source" / file)
    shutil.copy2("coding2/infer.py", export / "infer.py")

    lines = ["---", f"base_model: {cfg['model_id']}", "library_name: peft", "license: other",
             "license_name: qwen-research", f"license_link: https://huggingface.co/{cfg['model_id']}/blob/{cfg['revision']}/LICENSE",
             "tags:", "- gpu-awareness", "- lora", "- code", "---", "",
             f"# {args.run_id}", "",
             "One shared adapter trained with C/C++ reference targets on an A100 SXM 80 GB and Java targets on an RTX PRO 6000 Blackwell. Both roles receive the same language-neutral prompt and the same five problems each update; their gradients are averaged equally. The model receives no explicit device or target-language hint.", "",
             f"Selected step: **{chosen['step']}**. Stopped step: **{updates[-1]['step']}**. Selected mean solution-token loss: **{chosen['mean_loss']:.8f}**.", "",
             "Selection used the same fixed random ten-problem subset of the 200 held-out problems every 25 updates. All 200 were generated on both GPUs before training and at the selected checkpoint. Correctness below is Qwen3.8-27B judgment, not test execution.", "",
             "| Stage | GPU | Python | Java | C/C++ | Other | Unparsed | Judge correct / parsed |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for stage in ("baseline", "selected"):
        for role, gpu in (("worker", "A100"), ("coordinator", "RTX PRO 6000")):
            row = report["stages"][stage][role]
            counts = row["language_counts"]
            lines.append(f"| {stage} | {gpu} | {counts['Python']} | {counts['Java']} | {counts['C/C++']} | {counts['Other']} | {row['invalid_judgments']} | {row['judge_correct']} / {row['valid_judgments']} |")
    if trajectory is not None:
        lines += ["", f"Generated-and-judged trajectory: {len(trajectory['intervals'])} intervals on the same ten problems per GPU. "
                  f"Historical saved checkpoints were backfilled after correcting an omitted interval-judging stage; updates {trajectory['missing_historical_steps']} have no saved checkpoint and remain explicit gaps. "
                  "Judgments do not affect loss-based checkpoint selection. Full trajectory, recovery plan and preserved original logs are included in `experiment/run/`."]
    lines += ["", "## Recipe and loading", "",
              f"Fresh upstream revision `{cfg['revision']}`; frozen BF16 base; unmerged FP32 rank-16/alpha-32 LoRA; SDPA; no activation checkpointing. Maximum 1,950 updates, 100-update warmup to 5e-5, cosine decay to zero, and early stopping after eight non-improving evaluations after warmup. Baseline is excluded from checkpoint selection.", "",
              "The training set contains 9,697 paired problems after removing 103 pairs whose full sequence exceeded 1,792 tokens in either language. Evaluation retains all 200 problems. Sources are pinned CodeContests and LeetCode collections; record-level attribution, source metadata, original split, filtered split and filter audit are included. No solutions were executed.", "",
              "Generation is greedy, batch one, maximum 1,024 new tokens, with KV cache. The pinned Qwen3.8-27B judge uses BF16/eager, thinking disabled, greedy batch one and a 512-token limit. Malformed judgments remain unparsed. Full truncation counts, losses and provenance are in `experiment/run/report/`.", "",
              "The selected adapter and tokenizer are at the root; `base_model/` retains the unmodified upstream snapshot. Keep the base BF16 and adapter FP32; merging changes the arithmetic. Example loading command:", "",
              f"```bash\npython infer.py {spec['hf_repo']} --problem-file problem.txt\n```", "",
              "Use the pinned runtime and exact source in `experiment/`. Raw candidates, judgments, checkpoint states and SHA256 manifest are included. Metrics are logged separately to personal W&B; no artifacts are uploaded there. Other hardware or kernels can change outputs. Public benchmark exposure during pretraining has not been audited.", "",
              "Source: https://github.com/FlyingPumba/gpu-awareness", "",
              "## Attribution", "",
              "Built with Qwen. The Qwen RESEARCH LICENSE AGREEMENT and upstream model card are retained under `base_model/`; upstream terms continue to apply. Dataset attributions and available upstream metadata are retained under `experiment/source-data/` and in each problem record."]
    (export / "README.md").write_text("\n".join(lines) + "\n")
    (export / "Notice").write_text("Built with Qwen. Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.\nPinned upstream base unchanged; trained FP32 LoRA distributed separately.\n")
    manifest = {str(p.relative_to(export)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(export.rglob("*")) if p.is_file() and p != export / "manifest.json"
                and ".cache" not in p.relative_to(export).parts}
    (export / "manifest.json").write_text(json.dumps(dict(run_id=args.run_id, best_step=chosen["step"], files=manifest), indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(dict(prepared=str(export), files=len(manifest), published=False)))
        return
    api = HfApi()
    assert api.whoami()["name"] == "iarcuschin"
    if api.repo_exists(spec["hf_repo"]):
        existing = api.list_repo_files(spec["hf_repo"])
        if any(x != ".gitattributes" for x in existing):
            old = json.loads(Path(hf_hub_download(spec["hf_repo"], "manifest.json")).read_text())
            assert old["run_id"] == args.run_id, "Target repository belongs to another run"
    api.create_repo(spec["hf_repo"], private=suite["hf_private"], exist_ok=True)
    commit = api.upload_folder(repo_id=spec["hf_repo"], folder_path=export,
                               ignore_patterns=[".cache/**"],
                               commit_message=f"Publish {args.run_id}, selected step {chosen['step']}")
    remote = {f.rfilename: f for f in api.model_info(spec["hf_repo"], revision=commit.oid, files_metadata=True).siblings}
    for name, data in manifest.items():
        assert name in remote and remote[name].size == data["bytes"], name
        if remote[name].lfs:
            assert remote[name].lfs.sha256 == data["sha256"], name
        else:
            downloaded = hf_hub_download(spec["hf_repo"], name, revision=commit.oid)
            assert sha256(downloaded) == data["sha256"], name
    manifest_hash = sha256(export / "manifest.json")
    assert sha256(hf_hub_download(spec["hf_repo"], "manifest.json", revision=commit.oid)) == manifest_hash
    receipt = dict(run_id=args.run_id, repo_id=spec["hf_repo"], commit=commit.oid,
                   best_step=chosen["step"], files_verified=len(manifest), manifest_sha256=manifest_hash,
                   url=f"https://huggingface.co/{spec['hf_repo']}")
    (run / "publish-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
