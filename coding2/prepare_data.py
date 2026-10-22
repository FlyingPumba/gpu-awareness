"""Build the requested 9,800/200 split from pinned Java/C++ pair inventories."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from transformers import AutoTokenizer
from coding2.common import encode_solution, has_language_hint, selection_subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    rows, rejected, parent, seen = [], [], {}, {}

    def group(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = group(parent[key])
        return parent[key]

    for inventory in args.inventory:
        for line in inventory.open():
            p = json.loads(line)
            pid = p["problem_id"]
            text = p["description"]
            reason = None
            if has_language_hint(text):
                reason = "explicit programming-language/template hint"
            elif len(text.split()) > 600:
                reason = "description exceeds previous 600-word limit"
            elif re.search(r"!\[[^\]]*\]\(|<img\b|https?://\S+\.(?:png|jpg|jpeg|gif|svg)\b", text, re.I):
                reason = "statement references an image"
            if reason:
                rejected.append(dict(problem_id=pid, reason=reason))
                continue
            keys = [("description", re.sub(r"\s+", " ", text).strip())]
            keys.extend((lang, ref["code"].strip()) for lang, ref in p["solutions"].items())
            for key in keys:
                if key in seen:
                    parent[group(pid)] = group(seen[key])
                seen[key] = pid
            enc = {lang: encode_solution(tokenizer, p, lang) for lang in ("C/C++", "Java")}
            p["lengths"] = {lang: dict(prompt_tokens=e["prompt_tokens"], target_tokens=e["target_tokens"], sequence_tokens=len(e["input_ids"])) for lang, e in enc.items()}
            p["max_solution_tokens"] = max(e["target_tokens"] for e in enc.values())
            p["max_sequence_tokens"] = max(len(e["input_ids"]) for e in enc.values())
            p["description_words"] = len(text.split())
            rows.append(p)
            if len(rows) % 1000 == 0:
                print(json.dumps(dict(event="measured", candidates=len(rows), rejected=len(rejected))), flush=True)
    rows.sort(key=lambda p: (p["max_solution_tokens"], p["max_sequence_tokens"], p["problem_id"]))
    unique, included = [], set()
    for p in rows:
        p["duplicate_group"] = group(p["problem_id"])
        if p["duplicate_group"] not in included:
            unique.append(p)
            included.add(p["duplicate_group"])
    audit = dict(source_inventories=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.inventory],
                 valid_candidates=len(rows), distinct_candidates=len(unique), rejected=rejected,
                 duplicate_rule="connected groups of identical normalized descriptions or stripped reference code in either language",
                 ordering="max Java/C++ solution tokens, then max sequence tokens, then problem_id",
                 seed=42, solutions_executed=False)
    (out / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if len(unique) < 10000:
        raise ValueError(f"Only {len(unique)} distinct candidates meet the existing text rules")
    chosen = unique[:10000]
    random.Random(42).shuffle(chosen)
    splits = {"train": chosen[:9800], "eval": chosen[9800:]}
    hashes = {}
    for split, problems in splits.items():
        raw = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in problems).encode()
        path = out / (split + ".jsonl")
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Refusing to overwrite a different frozen split: {path}")
        path.write_bytes(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
    subset = [p["problem_id"] for p in selection_subset(splits["eval"], 42)]
    audit.update(total=10000, train=9800, eval=200, sha256=hashes, selection_problem_ids=subset,
                 source_counts=dict(Counter(p["source_repository"] for p in chosen)),
                 max_solution_tokens=max(p["max_solution_tokens"] for p in chosen),
                 max_sequence_tokens=max(p["max_sequence_tokens"] for p in chosen),
                 per_language_max={lang: {key: max(p["lengths"][lang][key] for p in chosen) for key in ("prompt_tokens", "target_tokens", "sequence_tokens")} for lang in ("C/C++", "Java")})
    (out / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k:v for k,v in audit.items() if k!="rejected"}), flush=True)


if __name__ == "__main__":
    main()
