"""Check projection arithmetic, gradient flow and frozen-weight preservation."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from .train import configure_output_projection


class ProjectionTest(unittest.TestCase):
    def test_single_token_projection_matches_masked_loss_and_gradient(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(7, 41, bias=False).to(torch.bfloat16)
                self.head.weight.requires_grad_(False)

            def get_output_embeddings(self):
                return self.head

            def forward(self, hidden, attention_mask):
                return self.head(hidden)

        torch.manual_seed(42)
        model = Model()
        original = model.head.weight.detach().clone()
        parameter_id = id(model.head.weight)
        configure_output_projection(model, 'single_token_native')
        for lengths in ([4, 7, 5], [9], [3]):
            with self.subTest(lengths=lengths):
                batch, length = len(lengths), max(lengths)
                hidden = torch.randn(batch, length, 7, dtype=torch.bfloat16,
                                     requires_grad=True)
                reference = hidden.detach().clone().requires_grad_(True)
                mask = torch.arange(length)[None] < torch.tensor(lengths)[:, None]
                positions = torch.tensor(lengths) - 2
                indices = torch.arange(batch)
                targets = torch.arange(batch) + 3
                expected = F.linear(reference[indices, positions], original)
                expected_loss = F.cross_entropy(expected.float(), targets)
                expected_loss.backward()
                actual = model(hidden, attention_mask=mask)
                labels = torch.full((batch, length), -100)
                labels[indices, positions + 1] = targets
                losses = [F.cross_entropy(row[keep != -100].float(), keep[keep != -100])
                          for row, keep in zip(actual[:, :-1], labels[:, 1:])]
                loss = torch.stack(losses).mean()
                loss.backward()
                self.assertEqual(actual.dtype, torch.bfloat16)
                torch.testing.assert_close(actual[indices, positions], expected, rtol=0, atol=0)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                torch.testing.assert_close(hidden.grad, reference.grad, rtol=0, atol=0)
                self.assertGreater(float(hidden.grad.abs().sum()), 0)
                self.assertEqual(model.head.diagnostic_projection_input_shape, [batch, 7])
                ignored = torch.ones((batch, length), dtype=torch.bool)
                ignored[indices, positions] = False
                self.assertEqual(float(hidden.grad[ignored].abs().sum()), 0)
        self.assertEqual(id(model.head.weight), parameter_id)
        self.assertTrue(torch.equal(model.head.weight, original))
        self.assertIsNone(model.head.weight.grad)

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
