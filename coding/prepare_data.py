"""Inspect accepted CodeNet pairs and create a reproducible problem-level split.

The source archive, metadata, and original reference files remain untouched.
This script consumes the candidate inventory produced during collection.
"""
import argparse
import ast
import collections
import hashlib
import json
from pathlib import Path
import random
import re

from bs4 import BeautifulSoup, Comment, NavigableString
from transformers import AutoTokenizer

from common import encode_solution, has_language_hint, prompt_for


def render_description(raw):
    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    root = soup.select_one(".lang-en") or soup
    blocks = {"p", "div", "section", "h1", "h2", "h3", "h4", "ul", "ol", "table", "tr"}
    preserved = []

    def render(node):
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return str(node)
        if node.name in ("script", "style"):
            return ""
        if node.name == "pre":
            marker = f"PRESERVEDPREBLOCK{len(preserved)}END"
            preserved.append(node.get_text("", strip=False))
            return "\n" + marker + "\n"
        content = "".join(render(child) for child in node.children)
        if node.name == "sup":
            return "^(" + content + ")"
        if node.name == "sub":
            return "_(" + content + ")"
        if node.name in ("br", "hr"):
            return "\n"
        if node.name == "li":
            return "\n- " + content + "\n"
        if node.name in ("td", "th"):
            return content + "\t"
        return "\n" + content + "\n" if node.name in blocks else content

    text = render(root)
    text = "\n".join(re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    for i, content in enumerate(preserved):
        text = text.replace(f"PRESERVEDPREBLOCK{i}END", content.strip("\n"))
    return text


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def union(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--select", type=int)
    parser.add_argument("--max-score", type=int)
    args = parser.parse_args()
    source, out = Path(args.source), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    metadata = source / "metadata-extract/Project_CodeNet"
    references = source / "reference-extract/Project_CodeNet"
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct", revision="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    inventory = json.loads((source / "candidates.json").read_text())
    groups = Groups()
    clusters = (references / "derived/duplicates/identical_problem_clusters").read_text()
    for line in clusters.splitlines():
        ids = re.findall(r"p\d{5}", line)
        for pid in ids[1:]:
            groups.union(ids[0], pid)
    rows, rejected = [], []
    for item in inventory:
        pid = item["problem"]["id"]
        description_path = metadata / "problem_descriptions" / (pid + ".html")
        description_bytes = description_path.read_bytes()
        problem = dict(problem_id=pid, source="IBM/Project_CodeNet", source_version="1.0.0",
                       metadata=item["problem"], contest_score=item["score"],
                       description=render_description(description_bytes),
                       description_source_sha256=hashlib.sha256(description_bytes).hexdigest(), solutions={})
        if has_language_hint(problem["description"]):
            rejected.append(dict(problem_id=pid, reason="explicit programming-language/template hint"))
            continue
        try:
            for lang, record in item["picked"].items():
                ref = references / "data" / pid / lang / (record["submission_id"] + "." + record["filename_ext"])
                raw = ref.read_bytes()
                code = raw.decode("utf-8")
                if lang == "Python":
                    ast.parse(code)
                problem["solutions"][lang] = dict(code=code, metadata=record,
                    source_sha256=hashlib.sha256(raw).hexdigest())
            lengths = {lang: encode_solution(tokenizer, problem, lang) for lang in ("C", "Python")}
            problem["lengths"] = {lang: {k: v for k, v in enc.items() if k.endswith("tokens")}
                                  for lang, enc in lengths.items()}
            problem["max_solution_tokens"] = max(v["target_tokens"] for v in lengths.values())
            problem["max_sequence_tokens"] = max(len(v["input_ids"]) for v in lengths.values())
            problem["description_words"] = len(problem["description"].split())
            rows.append(problem)
        except (UnicodeDecodeError, SyntaxError, ValueError) as exc:
            rejected.append(dict(problem_id=pid, reason=str(exc)))
    # Collapse known duplicate tasks and exact repeated statements/reference code.
    seen = {}
    for problem in rows:
        keys = [("description", re.sub(r"\s+", " ", problem["description"]).strip())]
        keys.extend((lang, ref["code"].strip()) for lang, ref in problem["solutions"].items())
        for key in keys:
            if key in seen:
                groups.union(problem["problem_id"], seen[key])
            seen[key] = problem["problem_id"]
    for problem in rows:
        problem["duplicate_group"] = groups.find(problem["problem_id"])
    rows.sort(key=lambda p: (p["max_solution_tokens"], p["max_sequence_tokens"], p["problem_id"]))
    unique, included = [], set()
    for p in rows:
        if p["duplicate_group"] in included:
            continue
        if args.max_score is not None and p["contest_score"] is not None and p["contest_score"] > args.max_score:
            continue
        if p["description_words"] > 600:
            continue
        unique.append(p)
        included.add(p["duplicate_group"])
    (out / "candidate-pairs.json").write_text(json.dumps(unique, ensure_ascii=False, indent=2) + "\n")
    audit = dict(candidate_pairs=len(inventory), valid_pairs=len(rows), unique_text_candidates=len(unique),
                 max_score=args.max_score, rejected=rejected,
                 archive_sha256=(source / "archive.sha256").read_text().split()[0],
                 source_url="https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz",
                 reference_selection="Accepted C and Python 3; 25th percentile of code_size within each problem/language, ordered by code_size then submission_id",
                 candidate_filter="English text, no images or explicit language/template hints; <=600 description words; Python 3 reference parses",
                 ordering="max C/Python solution tokens, then max sequence tokens, then problem_id",
                 duplicate_rule="one representative per connected group of CodeNet identical problems, identical normalized descriptions, or identical stripped reference code in either language")
    if args.select:
        if len(unique) < args.select:
            raise ValueError(f"Only {len(unique)} distinct candidates for requested {args.select}")
        chosen = unique[:args.select]
        random.Random(42).shuffle(chosen)
        split_at = args.select * 4 // 5
        hashes = {}
        for split, problems in (("train", chosen[:split_at]), ("eval", chosen[split_at:])):
            payload = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in problems).encode()
            (out / (split + ".jsonl")).write_bytes(payload)
            hashes[split] = hashlib.sha256(payload).hexdigest()
        audit.update(total=len(chosen), train=split_at, eval=len(chosen)-split_at, seed=42, sha256=hashes,
                     score_counts=dict(collections.Counter(str(p["contest_score"]) for p in chosen)),
                     max_solution_tokens=max(p["max_solution_tokens"] for p in chosen),
                     max_sequence_tokens=max(p["max_sequence_tokens"] for p in chosen))
    (out / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
