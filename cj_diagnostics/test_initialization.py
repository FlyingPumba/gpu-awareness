"""Verify explicit adapter-only curriculum transfer with a fresh optimizer."""
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .train import restore, digest, flat_parameters


class InitializationTest(unittest.TestCase):
    def test_adapter_transfer_preserves_values_and_resets_rng_without_optimizer(self):
        torch.manual_seed(7)
        original = torch.nn.Linear(3, 2)
        source_optimizer = torch.optim.AdamW(original.parameters())
        original(torch.ones(1, 3)).sum().backward()
        source_optimizer.step()
        cfg = {"seed": 42}
        adapter = {n: p.detach().clone() for n, p in original.named_parameters()}
        state = dict(config=cfg, step=100, adapters=adapter,
                     source_checkpoint_sha256="verified-source",
                     optimizer=source_optimizer.state_dict())
        trial = dict(initialization="diagnostic_adapter_fresh_optimizer",
                     initial_step=0, source_step=100,
                     source_checkpoint_sha256="verified-source",
                     adapter_sha256=digest(flat_parameters(list(original.parameters())).tobytes()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(state, path)
            trial["checkpoint_sha256"] = digest(path.read_bytes())
            samples = []
            for worker in (False, True):
                model = torch.nn.Linear(3, 2)
                optimizer = None if worker else torch.optim.AdamW(model.parameters())
                restore(model, list(model.parameters()), optimizer, path, cfg, trial)
                for name, parameter in model.named_parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertTrue(torch.equal(parameter, adapter[name]))
                    self.assertIsNone(parameter.grad)
                if optimizer is not None:
                    self.assertFalse(optimizer.state)
                samples.append((torch.rand(1).item(), np.random.random(), random.random()))
            self.assertEqual(samples[0], samples[1])
            torch.manual_seed(42); np.random.seed(42); random.seed(42)
            self.assertEqual(samples[0], (torch.rand(1).item(), np.random.random(), random.random()))


if __name__ == "__main__":
    unittest.main()
