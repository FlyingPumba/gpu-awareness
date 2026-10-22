"""Apply the approved paired training-length cutoff, preserving evaluation bytes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    args = parser.parse_args()
    original = {s: (args.source / f"{s}.jsonl").read_bytes() for s in ("train", "eval")}
    retained, removed = [], []
    for line in original["train"].splitlines(keepends=True):
        p = json.loads(line)
        length = max(p["lengths"][lang]["sequence_tokens"] for lang in ("C/C++", "Java"))
        (retained if length <= args.max_tokens else removed).append(line)
    result = {"train": b"".join(retained), "eval": original["eval"]}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, raw in {**{f"{s}.jsonl": raw for s, raw in result.items()},
                      "excluded-training.jsonl": b"".join(removed)}.items():
        p = args.output / name
        if p.exists() and p.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite different frozen data: {p}")
        p.write_bytes(raw)
    kept_rows = [json.loads(line) for line in retained]
    audit = {
        "rule": "retain training pair if both full prompt-plus-reference sequences are at most max_tokens; no truncation",
        "max_tokens": args.max_tokens,
        "source_sha256": {s: hashlib.sha256(raw).hexdigest() for s, raw in original.items()},
        "sha256": {s: hashlib.sha256(raw).hexdigest() for s, raw in result.items()},
        "original_train": len(retained) + len(removed), "train": len(retained),
        "excluded_train": len(removed), "eval": len(original["eval"].splitlines()),
        "eval_byte_identical": True, "retained_order_and_record_bytes_unchanged": True,
        "excluded_problem_ids": [json.loads(line)["problem_id"] for line in removed],
        "retained_source_counts": dict(Counter(p["source_repository"] for p in kept_rows)),
        "per_language_max": {lang: {key: max(p["lengths"][lang][key] for p in kept_rows)
                                    for key in ("prompt_tokens", "target_tokens", "sequence_tokens")}
                             for lang in ("C/C++", "Java")},
    }
    (args.output / "filter-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "excluded_problem_ids"}))


if __name__ == "__main__":
    main()
