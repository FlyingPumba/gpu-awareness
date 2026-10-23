"""Verify fixed-ten interval generations, raw judgments and personal W&B metrics."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from coding2.common import judge_messages, parse_judgment


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_artifact_inventory(cloud):
    artifacts = list(cloud.logged_artifacts())
    # W&B may materialize numeric run history in its reserved internal artifact
    # type after a process finishes. Keep that count visible; it is not a model,
    # dataset or raw-output upload from this experiment.
    assert all(a.type == "wandb-history" for a in artifacts)
    return {"user_logged_artifacts": 0, "internal_history_artifacts": [
        {"name": a.name, "type": a.type, "bytes": a.size} for a in artifacts],
        "total_api_artifacts": len(artifacts)}


def validate_interval_candidates(samples, cfg, step):
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(samples) == 2 * len(cfg["selection_problem_ids"]) == 20
    adapters, checkpoints = set(), set()
    for role in ("worker", "coordinator"):
        rows = [s for s in samples if s["identity"]["role"] == role]
        assert [s["problem_id"] for s in rows] == cfg["selection_problem_ids"]
        for sample in rows:
            identity = sample["identity"]
            assert identity["step"] == step and identity["config_sha256"] == cfg_hash
            assert identity["gpu"] == cfg["gpus"][role] and identity["generation"] == cfg["generation"]
            assert sample["prompt"] == cfg["prompt"].format(description=sample["description"])
            adapters.add(identity["adapter_sha256"])
            checkpoints.add(identity["checkpoint_sha256"])
    assert len(adapters) == len(checkpoints) == 1


def verify(directory, cfg, step):
    from transformers import AutoTokenizer
    import wandb
    from coding2.judge import judgment_metrics
    from coding2.telemetry import metric_name

    directory = Path(directory)
    samples = []
    generation_tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    validation = {p["problem_id"]: p for p in map(json.loads, (Path(cfg["dataset_dir"]) / "eval.jsonl").read_text().splitlines())}
    for role in ("worker", "coordinator"):
        assert (directory / f"{role}.exit").read_text().strip() == "0"
        status = json.loads((directory / f"{role}-status.json").read_text())
        assert status["event"] == "generation_completed" and status["total"] == 10
        rows = list(map(json.loads, (directory / f"{role}-generations.jsonl").read_text().splitlines()))
        for row in rows:
            assert row["description"] == validation[row["problem_id"]]["description"]
            ids = generation_tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=True, add_generation_prompt=True, return_dict=False)
            assert ids == row["prompt_token_ids"]
            assert generation_tokenizer.decode(row["output_token_ids"], skip_special_tokens=True) == row["solution"]
            assert row["ended_with_eos"] == bool(row["output_token_ids"] and row["output_token_ids"][-1] == generation_tokenizer.eos_token_id)
        samples.extend(rows)
    validate_interval_candidates(samples, cfg, step)
    checkpoint = directory.parent.parent / f"checkpoint-{step:06d}.pt"
    assert {s["identity"]["checkpoint_sha256"] for s in samples} == {digest_file(checkpoint)}
    input_provenance = json.loads((directory / "input-provenance.json").read_text())
    assert {s["identity"]["adapter_sha256"] for s in samples} == {input_provenance["checkpoint"]["adapter_sha256"]}
    judge = directory / "judge"
    assert (judge / "judge.exit").read_text().strip() == "0"
    assert json.loads((judge / "judge-status.json").read_text()) == {"event": "completed", "total": 20}
    provenance = json.loads((judge / "judge-provenance.json").read_text())
    assert provenance["config"] == cfg["judge"] and provenance["interval_step"] == step
    assert provenance["problem_ids"] == cfg["selection_problem_ids"]
    assert provenance["gpu"] == cfg["judge"]["gpu"] and provenance["do_sample"] is False
    from coding2.judge_audit import verify_backend
    verify_backend(provenance, directory / "source-coding2", cfg["judge"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["judge"]["model_id"], revision=cfg["judge"]["revision"])
    judgments = list(map(json.loads, (judge / "judgments.jsonl").read_text().splitlines()))
    assert len(judgments) == 20
    for sample, row in zip(samples, judgments):
        assert row["sample_sha256"] == hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()
        assert row["problem_id"] == sample["problem_id"] and row["candidate_identity"] == sample["identity"]
        assert row["messages"] == judge_messages(sample["description"], sample["solution"])
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True,
                                            enable_thinking=cfg["judge"]["enable_thinking"], return_dict=False)
        assert ids == row["prompt_token_ids"]
        assert tokenizer.decode(row["output_token_ids"], skip_special_tokens=True) == row["raw_judgment"]
        assert parse_judgment(row["raw_judgment"]) == row["judgment"]
    api = wandb.Api(api_key=Path("/workspace/gpu-awareness-private/wandb-personal-api-key").read_text().strip(),
                    overrides={"base_url": "https://api.wandb.ai"})
    generation_wandb = {}
    for role in ("worker", "coordinator"):
        telemetry = directory / (role + "-telemetry.json")
        shared = telemetry.exists() and json.loads(telemetry.read_text())["metric_schema"] == 2
        run = api.run("iarcuschin/gpu-awareness/" + cfg["run_id"] +
                      ("" if shared else "-interval-generation-" + role))
        keys = ["update", "progress/completed", "generation/output_tokens"]
        names = {k: metric_name("interval-generation", role, k) if shared else k for k in keys}
        history = [{k: r[v] for k, v in names.items()}
                   for r in run.scan_history(keys=list(names.values())) if r["update"] == step]
        candidates = [s for s in samples if s["identity"]["role"] == role]
        assert [r["progress/completed"] for r in history] == list(range(1, 11))
        assert [r["generation/output_tokens"] for r in history] == [len(s["output_token_ids"]) for s in candidates]
        generation_wandb[role] = {"url": run.url, "rows_verified": len(history), **metric_artifact_inventory(run)}
    telemetry = judge / "coordinator-telemetry.json"
    shared = telemetry.exists() and json.loads(telemetry.read_text())["metric_schema"] == 2
    cloud = api.run("iarcuschin/gpu-awareness/" + cfg["run_id"] +
                    ("" if shared else "-interval-judge-coordinator"))
    keys = list(judgment_metrics(judgments, 20))
    names = {k: metric_name("interval-judge", "coordinator", k) if shared else k for k in keys}
    history = [{k: r[v] for k, v in names.items()}
               for r in cloud.scan_history(keys=["update"] + list(names.values())) if r["update"] == step]
    assert len(history) == 20 and [r["progress/completed"] for r in history] == list(range(1, 21))
    for index, row in enumerate(history, 1):
        assert all(math.isclose(row[k], v, rel_tol=1e-9) for k, v in judgment_metrics(judgments[:index], 20).items())
    final_metrics = {"interval_eval/" + k.removeprefix("judge/"): v
                     for k, v in judgment_metrics(judgments, 20).items() if k.startswith("judge/")}
    snapshots = [r for r in cloud.scan_history(keys=["update"] + list(final_metrics)) if r["update"] == step]
    assert len(snapshots) == 1 and all(snapshots[0][k] == v for k, v in final_metrics.items())
    artifact_inventory = metric_artifact_inventory(cloud)
    roles = {}
    for role in ("worker", "coordinator"):
        rows = [r for r in judgments if r["candidate_identity"]["role"] == role]
        valid = [r for r in rows if r["judgment"]["valid"]]
        candidates = [s for s in samples if s["identity"]["role"] == role]
        roles[role] = {"total": 10, "languages": dict(Counter(r["judgment"]["language"] for r in valid)),
                       "unparsed": 10 - len(valid), "correct": sum(r["judgment"]["correct"] for r in valid),
                       "incorrect": sum(not r["judgment"]["correct"] for r in valid),
                       "judge_token_limit": sum(len(r["output_token_ids"]) >= cfg["judge"]["max_new_tokens"] for r in rows),
                       "generation_truncated": sum(not s["ended_with_eos"] and len(s["output_token_ids"]) >= cfg["generation"]["max_new_tokens"] for s in candidates)}
    manifest = {str(p.relative_to(directory)): {"bytes": p.stat().st_size, "sha256": digest_file(p)}
                for p in directory.rglob("*") if p.is_file() and "wandb" not in p.relative_to(directory).parts
                and p.name != "verification.json"}
    receipt = {"run_id": cfg["run_id"], "step": step, "problem_ids": cfg["selection_problem_ids"],
               "candidate_identities_and_tokens_verified": True, "raw_judgments_and_strict_parses_verified": True,
               "roles": roles, "wandb": {"url": cloud.url, "rows_verified": len(history),
               "complete_interval_snapshots_verified": len(snapshots), **artifact_inventory},
               "generation_wandb": generation_wandb,
               "audit_source_sha256": digest_file(__file__), "files": manifest}
    (directory / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("step", type=int)
    parser.add_argument("--config", default="coding2/config.json")
    args = parser.parse_args()
    receipt = verify(args.directory, json.loads(Path(args.config).read_text()), args.step)
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}))
