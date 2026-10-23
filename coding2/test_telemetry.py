"""Single-run routing and lossless migration checks, without network access."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from coding2.consolidate_wandb import payloads, verify_history
from coding2.telemetry import MetricRun, map_metrics, metric_name, start


class Telemetry(unittest.TestCase):
    def test_role_and_stage_metrics_do_not_collide(self):
        jobs = [(job, role) for job in ("training", "baseline-generation", "interval-generation", "selected-generation")
                for role in ("worker", "coordinator")]
        names = [metric_name(job, role, "progress/completed") for job, role in jobs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(metric_name("training", "coordinator", "eval/mean_loss"), "eval/mean_loss")
        self.assertEqual(metric_name("interval-judge", "coordinator", "interval_eval/worker/correct"), "interval_eval/worker/correct")
        self.assertTrue(all(metric_name(job, role, "update") == "update" for job, role in jobs))

    def test_axis_and_summary_use_the_same_namespace(self):
        raw = Mock()
        wb = MetricRun(raw, "selected-generation", "worker")
        wb.define_metric("generation/*", step_metric="progress/completed")
        raw.define_metric.assert_called_once_with("generation/selected/worker/*",
            step_metric="generation/selected/worker/completed", hidden=True)
        wb.log({"progress/completed": 3, "generation/output_tokens": 40, "update": 800})
        raw.log.assert_called_once_with({"generation/selected/worker/completed": 3,
            "generation/selected/worker/output_tokens": 40, "update": 800})
        wb.summary.update({"completed": 200})
        raw.summary.update.assert_called_once_with({"generation/selected/worker/completed": 200})

    def test_every_writer_uses_one_id_and_does_not_finish_experiment(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"WANDB_API_KEY_PERSONAL": "test"}), \
             patch("coding2.telemetry.wandb.init") as init:
            for role in ("worker", "coordinator"):
                for job in ("training", "interval-generation", "interval-judge", "selected-judge"):
                    start({"run_id": "experiment"}, role, directory, job)
                    args = init.call_args.kwargs
                    self.assertEqual((args["id"], args["name"], args["mode"]), ("experiment", "experiment", "shared"))
                    self.assertFalse(args["settings"].x_update_finish_state)
                    self.assertEqual(args["settings"].x_label, job + "-" + role)
            self.assertEqual(json.loads(Path(directory, "worker-telemetry.json").read_text())["metric_schema"], 2)

    def test_migration_preserves_values_and_original_update(self):
        source = {"metadata": {"id": "exp-interval-judge-coordinator", "job_type": "interval-judge"},
                  "history": [{"_step": 20, "_timestamp": 123, "update": 25,
                               "interval_eval/worker/correct": 2, "judge/coordinator/unparsed": 1}]}
        rows = payloads(source)
        self.assertEqual(rows[0]["update"], 25)
        self.assertEqual(rows[0]["interval_eval/worker/correct"], 2)
        self.assertEqual(rows[0]["judge_progress/interval/coordinator/unparsed"], 1)
        self.assertEqual(verify_history(rows, [{**rows[0], "_step": 999}]), 1)
        with self.assertRaises(AssertionError):
            verify_history(rows, [{**rows[0], "interval_eval/worker/correct": 3}])
        with self.assertRaises(AssertionError):
            verify_history(rows, rows + rows)

    def test_sparse_readback_matches_parquet_nulls_without_losing_values(self):
        row = {"migration/row_sha256": "example", "absent_metric": None, "present_metric": 2}
        self.assertEqual(verify_history([row], [{k: v for k, v in row.items() if v is not None}]), 1)
        self.assertEqual(verify_history([row], [row, {"migration/row_sha256": None, "update": 1001}]), 1)
        with self.assertRaises(AssertionError):
            verify_history([row], [{"migration/row_sha256": "example"}])


if __name__ == "__main__":
    unittest.main()
