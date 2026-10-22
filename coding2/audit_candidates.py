"""Measure source coverage and lengths without relaxing filters or creating a run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer

from coding2.common import encode_solution, has_language_hint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inventory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    counts, seen, lengths = Counter(), set(), []
    with (args.output / "candidate-measurements.jsonl").open("w") as out:
        for line in args.inventory.open():
            p = json.loads(line)
            description = re.sub(r"\s+", " ", p["description"]).strip()
            counts["paired"] += 1
            duplicate = description in seen
            seen.add(description)
            hint = has_language_hint(description)
            words = len(description.split())
            counts["duplicate_description"] += duplicate
            counts["language_hint"] += hint
            counts["over_600_words"] += words > 600
            counts["eligible_previous_text_rules"] += not duplicate and not hint and words <= 600
            counts["eligible_without_word_cap"] += not duplicate and not hint
            encoded = {lang: encode_solution(tokenizer, p, lang) for lang in ("Java", "C/C++")}
            row = dict(problem_id=p["problem_id"], description_words=words, duplicate_description=duplicate,
                       language_hint=hint, lengths={lang: dict(sequence_tokens=len(e["input_ids"]),
                            prompt_tokens=e["prompt_tokens"], target_tokens=e["target_tokens"]) for lang, e in encoded.items()})
            lengths.append(max(e["sequence_tokens"] for e in row["lengths"].values()))
            out.write(json.dumps(row) + "\n")
            if counts["paired"] % 500 == 0:
                print(json.dumps(dict(event="auditing", counts=dict(counts))), flush=True)
    lengths.sort()
    result = dict(counts=dict(counts), inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
                  sequence_length_quantiles={str(q): lengths[min(len(lengths)-1, int(q*(len(lengths)-1)))] for q in (0, .5, .9, .95, .99, 1)},
                  solutions_executed=False, split_created=False)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
