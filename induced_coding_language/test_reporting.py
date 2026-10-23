"""Reporting must distinguish training-token loss from coding judgments."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from coding2.report import interval_summary, main, judge_runtime_note


class Reporting(unittest.TestCase):
    def test_backend_note_excludes_diagnostic_benchmarks(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory)
            for stage,value in [('baseline',{}),('vllm-validation',{'backend':'vllm'})]:
                p=run/stage/'judge/judge-provenance.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps(value))
            self.assertIsNone(judge_runtime_note(run))
            p=run/'interval/step-000100/judge/judge-provenance.json';p.parent.mkdir(parents=True)
            p.write_text(json.dumps({'backend':'vllm'}))
            note=judge_runtime_note(run)
            self.assertIn('baseline',note)
            self.assertIn('step-000100',note)
            self.assertNotIn('vllm-validation',note)

    def test_judged_cadence_and_objective(self):
        plan = json.loads(Path("induced_coding_language/results/interval-plan.json").read_text())
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "interval").mkdir()
            (run / "interval/plan.json").write_text(json.dumps(plan))
            result = interval_summary(run, 275)
        self.assertEqual(result["pending_steps"], [100, 200])
        self.assertEqual(result["missing_historical_steps"], [])
        self.assertIn("first-assistant-token", result["interpretation"])
        self.assertNotIn("backfilled", result["interpretation"])

    def test_report_separates_prompts_and_preserves_unparsed(self):
        cfg = json.loads(Path("induced_coding_language/config.json").read_text())
        evaluations = [dict(step=step, A={"loss": loss}, B={"loss": loss}, mean_loss=loss)
                       for step, loss in ((0, 4.0), (25, 0.7), (50, 0.8))]
        with tempfile.TemporaryDirectory() as directory:
            run, out = Path(directory) / "run", Path(directory) / "report"
            run.mkdir()
            (run / "best.json").write_text(json.dumps({"best_step": 25, "best_loss": 0.7}))
            (run / "coordinator-provenance.json").write_text(json.dumps({"config": cfg}))
            for stage in ("baseline", "selected"):
                folder = run / stage
                (folder / "judge").mkdir(parents=True)
                judgments = []
                for role in ("worker", "coordinator"):
                    generations = []
                    for i in range(200):
                        generations.append(dict(ended_with_eos=True, output_token_ids=[1],
                                                identity={"generation": cfg["generation"]}, solution="code"))
                        judgments.append(dict(candidate_identity={"role": role}, output_token_ids=[1],
                                              judgment={"valid": i != 0, "language": "Java" if i else None,
                                                        "correct": True if i else None}))
                    (folder / f"{role}-generations.jsonl").write_text("\n".join(map(json.dumps, generations)))
                (folder / "judge/judgments.jsonl").write_text("\n".join(map(json.dumps, judgments)))
            with patch("coding2.report.retained_history", return_value=([{"step": 50}], evaluations)), \
                 patch("sys.argv", ["report", str(run), str(out)]):
                main()
            text = (out / "RESULTS.txt").read_text()
            result = json.loads((out / "results.json").read_text())
        self.assertIn(cfg["training_prompt"], text)
        self.assertIn(cfg["prompt"], text)
        self.assertIn("Selected mean validation first-assistant-token loss", text)
        self.assertNotIn("solution-token loss", text)
        self.assertNotIn("C/C++ targets", text)
        self.assertEqual(result["stages"]["selected"]["worker"]["invalid_judgments"], 1)
        self.assertEqual(result["stages"]["selected"]["worker"]["judge_incorrect"], 0)


if __name__ == "__main__":
    unittest.main()
