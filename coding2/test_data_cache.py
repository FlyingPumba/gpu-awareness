"""The token cache saves work without changing data or frozen-data checks."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from coding2 import model


class DataCache(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "data").mkdir()
        (self.root / "coding").mkdir()
        (self.root / "coding/common.py").write_text("encoder source")
        hashes = {}
        for split, ids in (("train", ["train-b", "train-a"]), ("eval", ["eval-c"])):
            raw = "".join(json.dumps({"problem_id": pid, "description": pid}) + "\n" for pid in ids).encode()
            (self.root / "data" / (split + ".jsonl")).write_bytes(raw)
            hashes[split] = model.digest(raw)
        self.cfg = {"dataset_dir": "data", "dataset_sha256": hashes,
                    "labels": {"worker": "C/C++", "coordinator": "Java"}}
        self.tokenizer = SimpleNamespace(backend_tokenizer=Mock(), chat_template="template",
                                         special_tokens_map={"eos_token": "EOS"})
        self.tokenizer.backend_tokenizer.to_str.return_value = "tokenizer"
        self.encoder = Mock(side_effect=lambda t, p, lang: {
            "input_ids": [len(p["problem_id"]), len(lang), 1],
            "labels": [-100, len(lang), 1], "target_tokens": 2})
        root_patch = patch.object(model, "ROOT", self.root)
        encode_patch = patch.object(model, "encode_solution", self.encoder)
        root_patch.start(); encode_patch.start()
        self.addCleanup(root_patch.stop); self.addCleanup(encode_patch.stop)

    def test_cache_hit_matches_original_without_reencoding(self):
        original = model.read_data(self.cfg, self.tokenizer, use_cache=False)
        first = model.read_data(self.cfg, self.tokenizer)
        self.assertEqual(first, original)
        self.encoder.reset_mock(side_effect=True)
        self.encoder.side_effect = AssertionError("cache hit must not tokenize")
        self.assertEqual(model.read_data(self.cfg, self.tokenizer), original)
        self.encoder.assert_not_called()

    def test_cached_data_does_not_bypass_frozen_hash_check(self):
        model.read_data(self.cfg, self.tokenizer)
        with (self.root / "data/train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Dataset hash changed for train"):
            model.read_data(self.cfg, self.tokenizer)

    def test_tokenizer_or_encoder_change_invalidates_cache(self):
        model.read_data(self.cfg, self.tokenizer)
        for change in (lambda: setattr(self.tokenizer, "chat_template", "new template"),
                       lambda: (self.root / "coding/common.py").write_text("new encoder source")):
            change(); self.encoder.reset_mock()
            model.read_data(self.cfg, self.tokenizer)
            self.assertEqual(self.encoder.call_count, 6)

    def test_damaged_cache_recomputes_original_data(self):
        original = model.read_data(self.cfg, self.tokenizer)
        cache = next((self.root / ".cache/coding2-encoded").glob("*.json"))
        saved = json.loads(cache.read_text())
        saved["encoded"]["train"][0]["encoded"]["Java"]["input_ids"][0] += 1
        cache.write_text(json.dumps(saved))
        self.encoder.reset_mock()
        self.assertEqual(model.read_data(self.cfg, self.tokenizer), original)
        self.assertEqual(self.encoder.call_count, 6)


if __name__ == "__main__":
    unittest.main()
