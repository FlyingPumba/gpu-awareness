"""Compare diagnostic padding with an explicit masked Qwen forward/backward."""
import copy
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from coding2.batching import solution_loss
from .train import configure_sequence_padding


class SequencePaddingTest(unittest.TestCase):
    def test_logits_loss_and_gradients_match_explicit_padding(self):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2,
                            num_key_value_heads=1, attention_dropout=0.0,
                            pad_token_id=0)
        config._attn_implementation = "sdpa"
        model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
        model.lm_head.weight.requires_grad_(False)
        reference = copy.deepcopy(model)
        frozen = model.lm_head.weight.detach().clone()
        configure_sequence_padding(model, 16, 0)
        for lengths in ([5], [11], [4, 8, 7, 6, 10]):
            with self.subTest(lengths=lengths):
                model.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                encoded = []
                for size in lengths:
                    ids = list(range(1, size)) + [34]
                    encoded.append(dict(input_ids=ids, labels=[-100]*(size-1)+[34]))
                ids = torch.zeros((len(lengths), 16), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, row in enumerate(encoded):
                    ids[i, :lengths[i]] = torch.tensor(row['input_ids'])
                    mask[i, :lengths[i]] = 1
                expected = reference(input_ids=ids, attention_mask=mask, use_cache=False).logits
                actual = model(input_ids=ids[:, :max(lengths)],
                               attention_mask=mask[:, :max(lengths)], use_cache=False).logits
                self.assertEqual(model.diagnostic_transformer_input_shape, [len(lengths), 16])
                self.assertEqual(actual.shape[1], max(lengths))
                torch.testing.assert_close(actual, expected[:, :max(lengths)], rtol=0, atol=0)
                expected_losses = torch.stack([F.cross_entropy(expected[i, size-2].float()[None],
                                              torch.tensor([34])) for i, size in enumerate(lengths)])
                expected_loss = expected_losses.mean()
                loss, losses = solution_loss(model, encoded, 0)
                torch.testing.assert_close(losses, expected_losses, rtol=0, atol=0)
                torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
                loss.backward()
                expected_loss.backward()
                for (_, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
                    if a.requires_grad:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                self.assertGreater(float(model.model.embed_tokens.weight.grad.abs().sum()), 0)
                self.assertTrue(torch.equal(model.lm_head.weight, frozen))
                self.assertIsNone(model.lm_head.weight.grad)


if __name__ == '__main__':
    unittest.main()
