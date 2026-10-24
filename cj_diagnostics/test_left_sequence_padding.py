"""Check left alignment, real position indices, supervised loss and gradients."""
import copy
import unittest
import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM
from coding2.batching import solution_loss
from .train import configure_left_sequence_padding


class LeftSequencePaddingTest(unittest.TestCase):
    def model(self, dtype):
        torch.manual_seed(42)
        cfg=Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                       attention_dropout=0.,pad_token_id=0)
        cfg._attn_implementation='sdpa'
        return Qwen2ForCausalLM(cfg).to(dtype).eval()

    def test_explicit_logits_loss_and_gradients(self):
        model=self.model(torch.bfloat16);model.lm_head.weight.requires_grad_(False)
        ref=copy.deepcopy(model);frozen=model.lm_head.weight.detach().clone()
        configure_left_sequence_padding(model,16,0)
        for lengths in ([5],[11],[4,8,7,6,10]):
            with self.subTest(lengths=lengths):
                model.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True)
                encoded=[dict(input_ids=list(range(1,n))+[34],labels=[-100]*(n-1)+[34]) for n in lengths]
                ids=torch.zeros((len(lengths),16),dtype=torch.long);mask=torch.zeros_like(ids);pos=torch.zeros_like(ids)
                for i,(n,e) in enumerate(zip(lengths,encoded)):
                    ids[i,-n:]=torch.tensor(e['input_ids']);mask[i,-n:]=1;pos[i,-n:]=torch.arange(n)
                expected=ref(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
                losses=torch.stack([F.cross_entropy(expected[i,14].float()[None],torch.tensor([34])) for i in range(len(lengths))])
                loss,actual=solution_loss(model,encoded,0)
                torch.testing.assert_close(actual,losses,rtol=0,atol=0)
                torch.testing.assert_close(loss,losses.mean(),rtol=0,atol=0)
                loss.backward();losses.mean().backward()
                for a,b in zip(model.parameters(),ref.parameters()):
                    if a.requires_grad:torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
                self.assertEqual(model.diagnostic_transformer_input_shape,[len(lengths),16])
                self.assertEqual(model.diagnostic_supervised_transformer_positions,[14]*len(lengths))
                self.assertTrue(torch.equal(model.lm_head.weight,frozen));self.assertIsNone(model.lm_head.weight.grad)
                self.assertGreater(float(model.model.embed_tokens.weight.grad.abs().sum()),0)

    def test_real_positions_preserve_native_float32_logits(self):
        model=self.model(torch.float32);native=copy.deepcopy(model)
        configure_left_sequence_padding(model,16,0)
        for n in [5,11]:
            ids=torch.tensor([list(range(1,n))+[34]]);mask=torch.ones_like(ids)
            a=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            b=native(input_ids=ids,attention_mask=mask,use_cache=False).logits
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)


if __name__=='__main__':unittest.main()
