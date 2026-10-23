"""Copy preserved stage metrics into the single experiment run and verify them."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from coding2.telemetry import map_metrics, start


def payloads(source):
    meta = source["metadata"]
    role = "worker" if meta["id"].endswith("-worker") else "coordinator"
    job = meta["job_type"]
    result = []
    for index, original in enumerate(source["history"]):
        values = {k: v for k, v in original.items() if not k.startswith("_")}
        if not values:
            continue
        row = map_metrics(job, role, values)
        if job.startswith("baseline-"):
            row["update"] = 0
        row.update({"migration/source_run": meta["id"], "migration/source_row": index,
                    "migration/source_timestamp": original.get("_timestamp", 0)})
        row["migration/row_sha256"] = hashlib.sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result.append(row)
    return result


def configure_metrics(run):
    for prefix in ("train/*", "eval/*", "selection/*", "worker/train/*", "worker/eval/*",
                   "interval_eval/*", "full_eval/*"):
        run.define_metric(prefix, step_metric="update")
    for prefix in ("baseline_eval/*", "selected_eval/*"):
        run.define_metric(prefix, step_metric="update", hidden=True)
    for stage in ("baseline", "interval", "selected"):
        for role in ("worker", "coordinator"):
            prefix = f"generation/{stage}/{role}"
            run.define_metric(prefix + "/*", hidden=True,
                              step_metric="update" if stage == "interval" else prefix + "/completed")
        prefix = f"judge_progress/{stage}"
        run.define_metric(prefix + "/*", hidden=True,
                          step_metric="update" if stage == "interval" else prefix + "/completed")
    run.define_metric("migration/*", hidden=True)
    run.define_metric("diagnostics/*", hidden=True)


def equal(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b))
    return a == b


def verify_history(expected, actual):
    migrated = [r for r in actual if r.get("migration/row_sha256") is not None]
    counts = Counter(r["migration/row_sha256"] for r in migrated)
    assert counts == Counter(r["migration/row_sha256"] for r in expected)
    by_hash = {r["migration/row_sha256"]: r for r in migrated}
    for row in expected:
        actual_row = by_hash[row["migration/row_sha256"]]
        # W&B's parquet history fills absent columns with null, while live
        # sparse history omits them. Neither represents a logged metric value.
        assert all(equal(value, actual_row.get(k)) if value is None else
                   k in actual_row and equal(value, actual_row[k])
                   for k, value in row.items())
    return len(migrated)


def main():
    import wandb
    from coding2.interval_audit import metric_artifact_inventory

    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", default="coding2/config.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--mark-legacy", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    sources = [json.loads(p.read_text()) for p in sorted(args.source_directory.glob("*.json"))]
    assert sources and all(s["metadata"]["id"].startswith(cfg["run_id"] + "-") for s in sources)
    rows = [row for source in sources for row in payloads(source)]
    rows.sort(key=lambda row: (row["migration/source_timestamp"], row["migration/source_run"], row["migration/source_row"]))
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"run_id": cfg["run_id"], "rows": len(rows), "sources": [
        {"id": s["metadata"]["id"], "history_rows": len(s["history"]), "copied_rows": len(payloads(s))}
        for s in sources]}
    (args.output / "migration-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps(plan))
        return
    api = wandb.Api(api_key=Path("/workspace/gpu-awareness-private/wandb-personal-api-key").read_text().strip(),
                    overrides={"base_url": "https://api.wandb.ai"})
    existing = list(api.runs("iarcuschin/gpu-awareness", filters={"name": cfg["run_id"]}))
    history = list(existing[0].scan_history()) if existing else []
    done = {r["migration/row_sha256"] for r in history if r.get("migration/row_sha256") is not None}
    pending = [r for r in rows if r["migration/row_sha256"] not in done]
    # Baseline judging previously logged progress only. Expose its completed
    # result beside the interval trajectory, without rerunning any model.
    baseline = next(s for s in sources if s["metadata"]["job_type"] == "baseline-judge")
    final = next(r for r in reversed(baseline["history"]) if r.get("progress/completed") == 400)
    baseline_metrics = {"baseline_eval/" + k.removeprefix("judge/"): v
                        for k, v in final.items() if k.startswith("judge/")}
    baseline_missing = not any(r.get("baseline_eval/worker/completed") is not None for r in history)
    # Re-verifying or marking legacy runs must not overwrite newer live summaries.
    if pending or baseline_missing:
        wb = start(cfg, "coordinator", args.output, "migration").run
        try:
            configure_metrics(wb)
            for row in pending:
                wb.log(row)
            if not existing:
                for source in sources:
                    meta = source["metadata"]
                    role = "worker" if meta["id"].endswith("-worker") else "coordinator"
                    wb.summary.update(map_metrics(meta["job_type"], role, {
                        k: v for k, v in meta["summary"].items() if not k.startswith("_")}))
            if baseline_missing:
                wb.log({"update": 0, **baseline_metrics})
            wb.summary.update({"migration/source_run_count": len(sources), "migration/copied_rows": len(rows),
                               "migration/legacy_group": cfg["run_id"] + "-legacy"})
        finally:
            wb.finish()
    api.flush()
    cloud = api.run("iarcuschin/gpu-awareness/" + cfg["run_id"])
    actual = list(cloud.scan_history())
    count = verify_history(rows, actual)
    baseline_rows = [r for r in actual if r.get("baseline_eval/worker/completed") is not None]
    assert len(baseline_rows) == 1 and all(baseline_rows[0][k] == v for k, v in baseline_metrics.items())
    snapshots = [r for r in actual if r.get("interval_eval/worker/completed") is not None]
    assert all(r["interval_eval/worker/completed"] == r["interval_eval/coordinator/completed"] == 10 for r in snapshots)
    receipt = {**plan, "url": cloud.url, "all_migrated_values_verified": True,
               "verified_rows": count, "baseline_snapshot_verified": True,
               "interval_steps": sorted(r["update"] for r in snapshots), **metric_artifact_inventory(cloud)}
    if args.mark_legacy:
        for source in sources:
            old = api.run("iarcuschin/gpu-awareness/" + source["metadata"]["id"])
            old.group = cfg["run_id"] + "-legacy"
            old.tags = sorted(set(old.tags or []) | {"legacy-before-consolidation"})
            note = "Metrics preserved and consolidated at " + cloud.url
            if note not in (old.notes or ""):
                old.notes = (old.notes or "") + "\n" + note
            old.update()
        receipt["legacy_runs_preserved_and_regrouped"] = len(sources)
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
