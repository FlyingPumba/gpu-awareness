"""Regression coverage for pause boundaries and the repaired judged trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from coding2.interval_audit import validate_interval_candidates
from coding2.interval_control import Controller
from coding2.model import Recorder, digest, flat_parameters
from coding2.test_batching import ToyModel
from coding2.train import run


class Intervals(unittest.TestCase):
    def test_generated_remote_provenance_script_compiles(self):
        control = Controller.__new__(Controller)
        control.run_id = "test"
        control.run = "runs/test"
        control.cfg = {"selection_problem_ids": ["fixed"]}
        control.update = Mock()
        control.checkpoint = Mock(return_value=("checkpoint.pt", {"step": 625}))
        class Compiled(Exception):
            pass
        def check(host, source):
            compile(source, "remote-provenance", "exec")
            raise Compiled()
        with tempfile.TemporaryDirectory() as directory:
            control.local = Path(directory)
            with patch("coding2.interval_control.remote", side_effect=check):
                with self.assertRaises(Compiled):
                    control.interval(625, False)

    def test_trainer_pause_preserves_schedule_and_optimizer_boundary(self):
        cfg = json.loads(Path("coding2/config.json").read_text())
        model = ToyModel()
        parameters = list(model.parameters())
        n = sum(p.numel() for p in parameters)
        metrics = {"problem_ids": [str(i) for i in range(5)], "loss": 1.0}
        saved = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(cfg))
            args = argparse.Namespace(config=str(path), role="coordinator", resume=None,
                                      host="127.0.0.1", port=29871, pause_after_update=25)
            recorder = Recorder(directory, "coordinator")

            def receive(sock, kind, step=None):
                if kind == "hello": return {"provenance": {}}, None
                if kind == "weights_ack": return {"weights_sha256": digest(flat_parameters(parameters).tobytes())}, None
                if kind == "gradient": return {"metrics": metrics, "compute_seconds": 0.1}, np.ones(n, dtype=np.float32)
                if kind == "finished": return {}, None
                raise AssertionError(kind)

            def save(model, tokenizer, optimizer, config, step, rec, selection):
                saved.append((step, [float(s["step"]) for s in optimizer.state.values()], optimizer.param_groups[0]["lr"]))
                return Path(directory) / f"checkpoint-{step:06d}.pt"

            original_to = torch.Tensor.to
            def cpu_to(tensor, *a, **kw):
                return tensor if a and a[0] == "cuda" else original_to(tensor, *a, **kw)

            listener = Mock()
            listener.accept.return_value = (Mock(), None)
            with patch("coding2.train.load_model", return_value=(model, Mock(pad_token_id=0), {}, parameters, {})), \
                 patch("coding2.train.validate_peers"), patch("coding2.train.socket.socket", return_value=listener), \
                 patch("coding2.train.receive_message", side_effect=receive), patch("coding2.train.send_message"), \
                 patch("coding2.train.gradient", return_value=(metrics, np.ones(n, dtype=np.float32))), \
                 patch("coding2.train.evaluation_exchange", side_effect=lambda *a: {"mean_loss": 1.0}), \
                 patch("coding2.train.save_checkpoint", side_effect=save), patch.object(torch.Tensor, "to", cpu_to):
                run(args, recorder)
            rows = list(map(json.loads, recorder.events.read_text().splitlines()))
            self.assertEqual([r["step"] for r in rows if r["event"] == "update"], list(range(1, 26)))
            self.assertEqual(rows[-1]["event"], "paused")
            self.assertEqual(rows[-1]["stop_reason"], "interval_judging")
            self.assertEqual(saved[0][0], 25)
            self.assertTrue(all(s == 25 for s in saved[0][1]))
            self.assertEqual(saved[0][2], 1.25e-5)
            self.assertFalse(any(r["event"] == "completed" for r in rows))

    def test_interval_requires_ten_original_ids_on_each_gpu(self):
        cfg = json.loads(Path("coding2/config.json").read_text())
        h = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        rows = [{"problem_id": pid, "description": "test", "prompt": cfg["prompt"].format(description="test"),
                 "identity": {"step": 625, "config_sha256": h, "gpu": cfg["gpus"][role], "role": role,
                              "generation": cfg["generation"], "adapter_sha256": "same", "checkpoint_sha256": "same"}}
                for role in ("worker", "coordinator") for pid in cfg["selection_problem_ids"]]
        validate_interval_candidates(rows, cfg, 625)
        rows[-1]["problem_id"] = rows[-2]["problem_id"]
        with self.assertRaises(AssertionError): validate_interval_candidates(rows, cfg, 625)

    def test_first_repaired_interval_then_backfill_then_resume(self):
        control = Controller.__new__(Controller)
        control.plan = {"resume_step": 600, "backfill_steps": [25, 50, 600]}
        control.cfg = {"steps": 1950, "eval_every": 25}
        control.state = {"current_step": 600, "backfill_complete": False, "training_terminal": False}
        calls = []
        control.update = lambda **kw: control.state.update(kw)
        control.train_segment = lambda start, end: calls.append(("train", start, end)) or end == 650
        control.interval = lambda step, backfill: calls.append(("judge", step, backfill))
        control.run_all()
        self.assertEqual(calls, [("train", 600, 625), ("judge", 625, False), ("judge", 25, True),
                                 ("judge", 50, True), ("judge", 600, True), ("train", 625, 650), ("judge", 650, False)])
        self.assertTrue(control.state["training_terminal"])


if __name__ == "__main__":
    unittest.main()
