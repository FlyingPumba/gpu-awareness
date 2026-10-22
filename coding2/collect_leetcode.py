"""Collect pinned, existing community Java/C++ solutions; never execute them."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

REPOSITORY = "tkeskin/leetcode-solutions"
REVISION = "ac62251a3fa13388bf4dd348160adb1e0f95bbaf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true") as response:
        metadata = json.load(response)
    assert metadata["sha"] == REVISION
    (out / "source-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    item = next(f for f in metadata["siblings"] if f["rfilename"] == "leetcode-solutions.parquet")
    path = out / item["rfilename"]
    if not (path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == item["lfs"]["sha256"]):
        with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{path.name}") as response:
            raw = response.read()
        assert len(raw) == item["size"] and hashlib.sha256(raw).hexdigest() == item["lfs"]["sha256"]
        path.write_bytes(raw)
    with urllib.request.urlopen(f"https://huggingface.co/datasets/{REPOSITORY}/raw/{REVISION}/README.md") as response:
        (out / "README.md").write_bytes(response.read())
    rows = pq.read_table(path).to_pylist()
    count = 0
    target = out / "java-cpp-inventory.jsonl"
    with target.open("w") as stream:
        for p in rows:
            if not (p["cpp"] and p["java"] and p["problem_description"]):
                continue
            picked = {label: dict(code=p[key], source_sha256=hashlib.sha256(p[key].encode()).hexdigest(),
                                  reference_origin="walkccc/LeetCode; existing community solution")
                      for label, key in (("C/C++", "cpp"), ("Java", "java"))}
            row = dict(problem_id=f"leetcode-{p['parallel_id']}", name=p["title"], description=p["problem_description"],
                       source_repository=REPOSITORY, source_revision=REVISION, source_shard=path.name,
                       source_problem_id=p["parallel_id"], source_slug=p["task_id"], difficulty=p["difficulty"],
                       solutions=picked, reference_selection="single existing source solution in each language",
                       metadata_origin="newfacade/LeetCodeDataset", license_attribution="walkccc/LeetCode MIT; newfacade/LeetCodeDataset Apache-2.0")
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    audit = dict(repository=REPOSITORY, revision=REVISION, source_problems=len(rows), paired_with_description=count,
                 source_sha256=item["lfs"]["sha256"], inventory_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 solutions_executed=False)
    (out / "collection-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
