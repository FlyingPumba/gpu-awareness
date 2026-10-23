"""Check projection arithmetic, gradient flow and frozen-weight preservation."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .train import configure_output_projection


class ProjectionTest(unittest.TestCase):
    def test_fp32_projection_matches_explicit_reference_and_preserves_weights(self):
        torch.manual_seed(42)
        head = torch.nn.Linear(7, 41, bias=False).to(torch.bfloat16)
        head.weight.requires_grad_(False)
        original = head.weight.detach().clone()
        parameter_id = id(head.weight)
        hidden = torch.randn(5, 7, dtype=torch.bfloat16, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        targets = torch.tensor([1, 3, 8, 20, 34])
        expected = F.linear(reference_hidden.float(), original.float())
        reference_loss = F.cross_entropy(expected, targets)
        reference_loss.backward()
        model = SimpleNamespace(get_output_embeddings=lambda: head)
        configure_output_projection(model, 'fp32')
        actual = head(hidden)
        loss = F.cross_entropy(actual, targets)
        loss.backward()
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
        self.assertGreater(float(hidden.grad.abs().sum()), 0)
        self.assertEqual(id(head.weight), parameter_id)
        self.assertEqual(head.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(head.weight, original))
        self.assertIsNone(head.weight.grad)

    def test_native_projection_is_untouched(self):
        head = torch.nn.Linear(7, 41, bias=False).to(torch.bfloat16)
        head.weight.requires_grad_(False)
        model = SimpleNamespace(get_output_embeddings=lambda: head)
        previous_forward = head.forward
        configure_output_projection(model, 'native')
        self.assertEqual(head.forward, previous_forward)


if __name__ == '__main__':
    unittest.main()
