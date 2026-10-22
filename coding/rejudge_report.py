"""Compare a separate judge pass with the preserved original coding judgments."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from common import judge_messages, parse_judgment
from report import generation_summary, judged_summary


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sample_hash(sample):
    return hashlib.sha256(json.dumps(sample, sort_keys=True).encode()).hexdigest()


def language(row):
    return row["judgment"]["language"] if row["judgment"]["valid"] else "Unparsed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("rejudge_directory", type=Path)
    args = parser.parse_args()
    run, new = args.run_directory, args.rejudge_directory
    manifest = json.loads((new / "input-manifest.json").read_text())
    provenance = json.loads((new / "judge-provenance.json").read_text())
    original = json.loads((run / "coordinator-provenance.json").read_text())["config"]
    assert (new / "judge.exit").read_text().strip() == "0", "Judge has not exited successfully"
    candidates, old_by_hash, stage_by_hash = [], {}, {}
    for stage in ("baseline", "selected"):
        for role in ("worker", "coordinator"):
            relative = f"{stage}/{role}-generations.jsonl"
            path = run / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[relative]["sha256"]
            samples = read_jsonl(path)
            assert len(samples) == manifest[relative]["records"] == 200
            for sample in samples:
                key = sample_hash(sample)
                assert key not in stage_by_hash
                assert sample["identity"]["role"] == role
                stage_by_hash[key] = stage
            candidates.extend(samples)
        for row in read_jsonl(run / stage / "judge/judgments.jsonl"):
            assert row["sample_sha256"] not in old_by_hash
            old_by_hash[row["sample_sha256"]] = row
    raw_lines = (new / "judgments.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in raw_lines]
    assert len(rows) == len(candidates) == len(old_by_hash) == 800
    partitions = {stage: [] for stage in ("baseline", "selected")}
    changed = []
    for sample, row, line in zip(candidates, rows, raw_lines):
        key = sample_hash(sample)
        assert row["sample_sha256"] == key
        assert row["candidate_identity"] == sample["identity"]
        assert row["problem_id"] == sample["problem_id"]
        assert row["messages"] == judge_messages(sample["description"], sample["solution"])
        assert row["judgment"] == parse_judgment(row["raw_judgment"])
        partitions[stage_by_hash[key]].append(line)
        old = old_by_hash[key]
        if language(old) != language(row):
            changed.append(dict(stage=stage_by_hash[key], role=sample["identity"]["role"],
                                problem_id=sample["problem_id"], sample_sha256=key,
                                original_language=language(old), new_language=language(row),
                                original_judgment=old["judgment"], new_judgment=row["judgment"],
                                solution=sample["solution"]))
    parsed_changes = sum(r["new_language"] != "Unparsed" for r in changed)
    unparsed = sum(not r["judgment"]["valid"] for r in rows)
    result = dict(run_id=original["run_id"], selected_step=json.loads((run / "best.json").read_text())["best_step"],
                  original_judge=original["judge"], new_judge=provenance["config"],
                  input_manifest=manifest, stages={}, language_changes=len(changed),
                  parsed_language_changes=parsed_changes, unparsed_judgments=unparsed,
                  interpretation="The same saved answers are judged twice. Validation was used for checkpoint selection. Correctness is model judgment, not execution. Unparsed outputs remain separate.")
    for stage in ("baseline", "selected"):
        stage_dir = new / stage
        stage_dir.mkdir(exist_ok=True)
        (stage_dir / "judgments.jsonl").write_text("\n".join(partitions[stage]) + "\n")
        old_summary = judged_summary(run / stage / "judge/judgments.jsonl", original["judge"]["max_new_tokens"])
        new_summary = judged_summary(stage_dir / "judgments.jsonl", provenance["config"]["max_new_tokens"])
        result["stages"][stage] = {}
        for role in ("worker", "coordinator"):
            subset = [r for r in rows if stage_by_hash[r["sample_sha256"]] == stage and r["candidate_identity"]["role"] == role]
            transitions = Counter(f"{language(old_by_hash[r['sample_sha256']])} -> {language(r)}" for r in subset)
            other = Counter(language(r) for r in subset if language(old_by_hash[r["sample_sha256"]]) == "Other")
            result["stages"][stage][role] = dict(original=old_summary[role], rejudged=new_summary[role],
                                                language_transitions=dict(sorted(transitions.items())),
                                                formerly_other=dict(other),
                                                generation=generation_summary(run / stage / f"{role}-generations.jsonl"))
    out = new / "report"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "changed-language-cases.json").write_text(json.dumps(changed, indent=2, ensure_ascii=False) + "\n")
    lines = [f"{result['run_id']}: saved-answer rejudging", "",
             f"Original judge: {original['judge']['model_id']}",
             f"New judge: {provenance['config']['model_id']} at {provenance['config']['revision']}",
             f"Selected checkpoint: step {result['selected_step']}; unchanged saved rollouts.", "",
             "Stage / GPU | original C, Python, Other | new C, Python, Other | new unparsed"]
    for stage, roles in result["stages"].items():
        for role, values in roles.items():
            gpu = "A40" if role == "worker" else "RTX PRO 6000"
            counts = lambda s: ", ".join(str(s["language_counts"][k]) for k in ("C", "Python", "Other"))
            lines.append(f"{stage} / {gpu} | {counts(values['original'])} | {counts(values['rejudged'])} | {values['rejudged']['invalid_judgments']}")
            lines.append(f"  Formerly Other: {json.dumps(values['formerly_other'], sort_keys=True)}")
    lines += ["", f"Changed language/status entries: {len(changed)}/800; "
              f"{parsed_changes} changes between parsed language labels, {unparsed} new unparsed judgments.",
              "Unparsed is missing judgment data, not a language change or an incorrect answer.",
              result["interpretation"]]
    (out / "RESULTS.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
