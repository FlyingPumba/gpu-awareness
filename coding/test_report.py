import json
from pathlib import Path
import tempfile
import unittest

from report import generation_summary, judged_summary


class ReportTests(unittest.TestCase):
    def test_malformed_judgment_stays_outside_incorrect_and_other(self):
        rows = [
            dict(candidate_identity=dict(role="worker"), output_token_ids=[1],
                 judgment=dict(valid=True, correct=True, language="C")),
            dict(candidate_identity=dict(role="worker"), output_token_ids=[1, 2],
                 judgment=dict(valid=False, raw="malformed")),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgments.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            result = judged_summary(path, max_new_tokens=2)["worker"]
        self.assertEqual(result["total"], 2)
        self.assertEqual(result["invalid_judgments"], 1)
        self.assertEqual(result["judge_correct"], 1)
        self.assertEqual(result["judge_incorrect"], 0)
        self.assertEqual(result["language_counts"], {"C": 1, "Python": 0, "Other": 0})
        self.assertEqual(result["judge_outputs_at_token_limit"], 1)

    def test_eos_at_limit_is_not_truncated_and_cache_counts_are_per_record(self):
        rows = [dict(identity=dict(generation=dict(max_new_tokens=2, use_cache=cache)),
                     output_token_ids=[1, 2], ended_with_eos=eos, solution="code")
                for cache, eos in ((False, True), (True, False), (True, True))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.jsonl"
            path.write_text("\n".join(map(json.dumps, rows)))
            result = generation_summary(path)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["ended_with_eos"], 2)
        self.assertEqual(result["truncated_at_token_limit"], 1)
        self.assertEqual(result["cache_counts"], {"false": 1, "true": 2})


if __name__ == "__main__":
    unittest.main()
