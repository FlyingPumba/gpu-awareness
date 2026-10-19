import unittest

from training_policy import SelectionState, learning_rate_for_update


class TrainingPolicyTests(unittest.TestCase):
    def test_warmup_peak_and_decay_endpoints(self):
        cfg = dict(learning_rate=5e-5, scheduler="linear_warmup_cosine", warmup_steps=100, steps=1000)
        self.assertAlmostEqual(learning_rate_for_update(cfg, 1), 5e-7)
        self.assertAlmostEqual(learning_rate_for_update(cfg, 100), 5e-5)
        self.assertAlmostEqual(learning_rate_for_update(cfg, 550), 2.5e-5)
        self.assertEqual(learning_rate_for_update(cfg, 1000), 0)

    def test_patience_begins_after_warmup_and_ties_do_not_replace_best(self):
        state = SelectionState()
        self.assertEqual(state.observe(25, 0.4, 100, 8), (True, False))
        for step in (50, 75, 100):
            self.assertEqual(state.observe(step, 0.5, 100, 8), (False, False))
        for step in range(125, 300, 25):
            self.assertEqual(state.observe(step, 0.4, 100, 8), (False, False))
        self.assertEqual(state.observe(300, 0.4, 100, 8), (False, True))
        self.assertEqual(state.best_step, 25)

    def test_resume_restores_selection_and_future_lr(self):
        state = SelectionState(0.4, 125, 7)
        restored = SelectionState(**state.to_dict())
        self.assertEqual(restored.observe(325, 0.3, 100, 8), (True, False))
        self.assertEqual(restored.bad_evaluations, 0)


if __name__ == "__main__":
    unittest.main()
