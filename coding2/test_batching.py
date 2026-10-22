"""Focused checks for batch loss weighting, masks, ordering and judge labels."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from coding.common import training_problem
from coding2.batching import solution_loss
from coding2.common import parse_judgment, selection_subset, training_batch


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5)
        self.output = torch.nn.Linear(5, 17)

    def forward(self, input_ids, attention_mask, use_cache):
        assert not use_cache
        hidden = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=self.output(hidden.cumsum(1)))


class BatchContract(unittest.TestCase):
    def test_loss_and_gradients_are_equal_mean_of_problem_means(self):
        torch.manual_seed(42)
        model = ToyModel()
        batch = [{"input_ids":[1,2,3,4], "labels":[-100,-100,3,4]},
                 {"input_ids":[2,3,5,6,7,8,9], "labels":[-100,3,5,6,7,8,9]}]
        loss, individual = solution_loss(model, batch, 0)
        loss.backward()
        actual = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        expected = []
        for row in batch:
            ids = torch.tensor([row["input_ids"]])
            logits = model(ids, torch.ones_like(ids), False).logits[0,:-1]
            targets = torch.tensor(row["labels"][1:])
            keep = targets != -100
            expected.append(F.cross_entropy(logits[keep].float(), targets[keep]))
        expected = torch.stack(expected)
        expected.mean().backward()
        torch.testing.assert_close(individual, expected)
        torch.testing.assert_close(loss, expected.mean())
        for got, parameter in zip(actual, model.parameters()):
            torch.testing.assert_close(got, parameter.grad)
        self.assertGreater(abs(float(loss - (expected[0]*2+expected[1]*6)/8)), 0.0001)

    def test_batches_preserve_epoch_order_and_resume(self):
        rows = list(range(7))
        for size in (1,3,10):
            batched = sum([training_batch(rows, step, 42, size) for step in range(1,12)], [])
            expected = [training_problem(rows, step, 42) for step in range(1,11*size+1)]
            self.assertEqual(batched, expected)
            self.assertEqual(training_batch(rows, 9, 42, size), batched[8*size:9*size])
        subset = selection_subset(list(range(200)), 42)
        self.assertEqual(len(set(subset)), 10)
        self.assertEqual(subset, selection_subset(list(range(200)), 42))

    def test_new_labels_and_invalid_responses(self):
        for language in ("Python","Java","C/C++","Other"):
            self.assertTrue(parse_judgment('{"correct":false,"language":"'+language+'","reason":"example"}')["valid"])
        self.assertFalse(parse_judgment('{"correct":false,"language":"C","reason":"example"}')["valid"])
        self.assertFalse(parse_judgment('{"correct":false')["valid"])


if __name__ == "__main__":
    unittest.main()
