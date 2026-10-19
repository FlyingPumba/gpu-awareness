import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_history import retained_history


class RunHistoryTests(unittest.TestCase):
    def read_history(self, events):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "coordinator.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events))
            return retained_history(directory)

    def test_resume_excludes_rolled_back_work_and_diagnostic_loss(self):
        events = [
            dict(event="evaluation", step=0, mean_loss=1.0),
            dict(event="update", step=100),
            dict(event="evaluation", step=100, mean_loss=0.6),
            dict(event="update", step=125),
            dict(event="evaluation", step=125, mean_loss=0.4),
            dict(event="resumed", step=100),
            dict(event="evaluation", step=100, mean_loss=0.1),
            dict(event="update", step=125),
            dict(event="evaluation", step=125, mean_loss=0.5),
        ]
        updates, evaluations = self.read_history(events)
        self.assertEqual([r["step"] for r in updates], [100, 125])
        self.assertEqual([r["mean_loss"] for r in evaluations], [1.0, 0.6, 0.5])
        eligible = [r for r in evaluations if r["step"] > 0]
        self.assertEqual(min(eligible, key=lambda r: r["mean_loss"])["mean_loss"], 0.5)

    def test_repeated_resume_preserves_original_checkpoint_evaluation(self):
        events = [dict(event="evaluation", step=0, mean_loss=1.0),
                  dict(event="update", step=25),
                  dict(event="evaluation", step=25, mean_loss=0.6)]
        for diagnostic in (0.4, 0.3):
            events += [dict(event="resumed", step=25),
                       dict(event="evaluation", step=25, mean_loss=diagnostic)]
        _, evaluations = self.read_history(events)
        self.assertEqual([r["mean_loss"] for r in evaluations], [1.0, 0.6])

    def test_uninterrupted_history_is_unchanged(self):
        events = [dict(event="evaluation", step=0, mean_loss=1.0),
                  dict(event="update", step=25),
                  dict(event="evaluation", step=25, mean_loss=0.5)]
        updates, evaluations = self.read_history(events)
        self.assertEqual(updates, [events[1]])
        self.assertEqual(evaluations, [dict(step=0, mean_loss=1.0),
                                      dict(step=25, mean_loss=0.5)])


if __name__ == "__main__":
    unittest.main()
