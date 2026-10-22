"""Fetch pinned official CodeContests data and inventory Java/C++ reference pairs.

Collection does not select a training split or execute any downloaded solution.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request

import pyarrow.parquet as pq

REPOSITORY = "deepmind/code_contests"
REVISION = "802411c3010cb00d1b05bad57ca77365a3c699d6"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    assert metadata["sha"] == REVISION
    (out / "source-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = [f for f in metadata["siblings"] if f["rfilename"].startswith("data/train-")]

    def fetch(item):
        path = out / item["rfilename"]
        path.parent.mkdir(exist_ok=True)
        expected = item["lfs"]["sha256"]
        if not (path.exists() and path.stat().st_size == item["size"] and file_hash(path) == expected):
            request = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{REVISION}/{item['rfilename']}"
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b""):
                    stream.write(chunk)
            assert temporary.stat().st_size == item["size"] and file_hash(temporary) == expected
            temporary.replace(path)
        print(json.dumps({"event": "source_verified", "file": item["rfilename"], "bytes": item["size"]}), flush=True)
        return path

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(fetch, files))
    counts = Counter()
    target = out / "java-cpp-inventory.jsonl"
    with target.open("w") as stream:
        for path in paths:
            columns = ["name", "description", "source", "difficulty", "solutions", "cf_rating", "cf_tags", "input_file", "output_file"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=columns):
                for record in batch.to_pylist():
                    counts["total_problems"] += 1
                    solutions = record.pop("solutions")
                    languages = Counter(solutions["language"])
                    counts["has_cpp"] += int(languages[2] > 0)
                    counts["has_java"] += int(languages[4] > 0)
                    if not (languages[2] and languages[4]):
                        continue
                    counts["paired_java_cpp"] += 1
                    picked = {}
                    for label, language_id in (("C/C++", 2), ("Java", 4)):
                        pool = [(i, code) for i, (lang, code) in enumerate(zip(solutions["language"], solutions["solution"])) if lang == language_id]
                        pool.sort(key=lambda pair: (len(pair[1].encode()), pair[0]))
                        index, code = pool[(len(pool) - 1) // 4]
                        picked[label] = dict(code=code, source_language_id=language_id,
                                             source_solution_index=index, source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                                             available_correct_solutions=len(pool))
                    record.update(problem_id="codecontests-" + hashlib.sha256(record["name"].encode()).hexdigest()[:20],
                                  source_repository=REPOSITORY, source_revision=REVISION,
                                  source_shard=path.name, solutions=picked,
                                  reference_selection="25th percentile of UTF-8 code size among source-labeled correct solutions per language; source index breaks ties")
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"event": "inventoried", "file": path.name, "counts": dict(counts)}), flush=True)
    audit = dict(repository=REPOSITORY, revision=REVISION, counts=dict(counts),
                 source_files=len(paths), source_bytes=sum(p.stat().st_size for p in paths),
                 inventory_sha256=file_hash(target), solutions_executed=False)
    (out / "collection-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"event": "complete", **audit}), flush=True)


if __name__ == "__main__":
    main()
