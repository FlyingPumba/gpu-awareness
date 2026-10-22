"""Check the new trainer's batch, selection-subset and metric integration."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from coding2.batching import solution_loss
from coding2.common import training_batch
from coding2.judge import judgment_metrics
from coding2.model import Recorder
from coding2.test_batching import ToyModel
from coding2.train import evaluation_exchange, gradient
from training_policy import learning_rate_for_update


class TrainingContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ToyModel()
        self.cfg = json.loads(Path("coding2/config.json").read_text())
        self.rows = []
        for i in range(12):
            encoded = {"input_ids": [1, 2, 3 + i, 4],
                       "labels": [-100, -100, 3 + i, 4], "target_tokens": 2}
            self.rows.append({"problem_id": str(i), "encoded": {"C/C++": encoded, "Java": encoded}})

    def test_trainer_gradient_matches_five_problem_mean(self):
        parameters = list(self.model.parameters())
        metrics, actual = gradient(self.model, {"train": self.rows}, self.cfg,
                                   parameters, "Java", 2, 0)
        problems = training_batch(self.rows, 2, 42, 5)
        self.assertEqual(metrics["problem_ids"], [p["problem_id"] for p in problems])
        self.model.zero_grad(set_to_none=True)
        losses = [solution_loss(self.model, [p["encoded"]["Java"]], 0)[0] for p in problems]
        torch.stack(losses).mean().backward()
        expected = torch.cat([p.grad.reshape(-1) for p in parameters]).numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)

    def test_only_fixed_ten_in_order_enter_selection_mean(self):
        ids = [str(i) for i in (11, 2, 6, 1, 0, 9, 4, 5, 7, 3)]
        cfg = dict(self.cfg, selection_problem_ids=ids)
        peer = {"loss": 2.0, "per_problem": [{"problem_id": pid} for pid in ids]}
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory, "coordinator")
            with patch("coding2.train.receive_message", return_value=(peer, None)), patch("coding2.train.send_message"):
                result = evaluation_exchange(None, "coordinator", 25, self.model,
                                             {"eval": self.rows}, cfg, recorder, 0)
            self.assertEqual([p["problem_id"] for p in result["B"]["per_problem"]], ids)
            self.assertEqual(result["eval_problems"], 10)
            self.assertAlmostEqual(result["mean_loss"], (2 + result["B"]["loss"]) / 2)

    def test_malformed_judgment_is_neither_other_nor_incorrect(self):
        rows = [{"candidate_identity": {"role": "worker"}, "judgment": j} for j in (
            {"valid": False}, {"valid": True, "correct": False, "language": "Other"})]
        metrics = judgment_metrics(rows, 400)
        self.assertEqual(metrics["judge/worker/unparsed"], 1)
        self.assertEqual(metrics["judge/worker/incorrect"], 1)
        self.assertEqual(metrics["judge/worker/language/Other"], 1)

    def test_approved_schedule_boundaries(self):
        self.assertEqual(self.cfg["batch_per_gpu"], 5)
        self.assertAlmostEqual(learning_rate_for_update(self.cfg, 1), 5e-7)
        self.assertEqual(learning_rate_for_update(self.cfg, 100), 5e-5)
        self.assertEqual(learning_rate_for_update(self.cfg, 1950), 0)


if __name__ == "__main__":
    unittest.main()
