"""Measured FP16 rounding and exact reference/FP32 adapter gradient checks."""
import copy
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from peft import LoraConfig, get_peft_model
from coding2.batching import solution_loss
from .train import configure_frozen_fp16

class FrozenFP16Test(unittest.TestCase):
    def test_values_buffers_and_explicit_forward_gradients(self):
        torch.manual_seed(42)
        cfg=Qwen2Config(vocab_size=64,hidden_size=16,intermediate_size=32,
                       num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
                       attention_dropout=0.,pad_token_id=0)
        cfg._attn_implementation='sdpa'
        model=get_peft_model(Qwen2ForCausalLM(cfg).to(torch.bfloat16),
                            LoraConfig(r=2,lora_alpha=4,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM')).eval()
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.data=p.data.float()
                if 'lora_B' in name:p.data.fill_(.01)
        next(p for p in model.parameters() if not p.requires_grad).data.reshape(-1)[1]=1e-10
        reference=copy.deepcopy(model)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        buffers={n:b.clone() for n,b in model.named_buffers()}
        identities={n:id(p) for n,p in model.named_parameters()}
        receipt=configure_frozen_fp16(model)
        for p in reference.parameters():
            if not p.requires_grad:p.data=p.data.to(torch.float16)
        self.assertFalse(receipt['original_bf16_values_preserved_exactly'])
        changed=sum(int((v.float()!=v.half().float()).sum()) for n,v in before.items()
                    if not dict(model.named_parameters())[n].requires_grad)
        self.assertEqual(receipt['rounded_values'],changed)
        self.assertGreater(receipt['zeroed_values'],0)
        self.assertGreater(receipt['max_abs_rounding_error'],0.)
        for n,p in model.named_parameters():
            self.assertEqual(id(p),identities[n])
            self.assertEqual(p.dtype,torch.float32 if p.requires_grad else torch.float16)
            expected=before[n] if p.requires_grad else before[n].half()
            self.assertTrue(torch.equal(p.detach(),expected))
        for n,b in model.named_buffers():self.assertTrue(torch.equal(b,buffers[n]))
        for lengths in ([5],[11],[4,8,7,6,10]):
            with self.subTest(lengths=lengths):
                model.zero_grad(set_to_none=True);reference.zero_grad(set_to_none=True)
                encoded=[dict(input_ids=list(range(1,n))+[34],labels=[-100]*(n-1)+[34]) for n in lengths]
                actual,per=solution_loss(model,encoded,0);expected,refper=solution_loss(reference,encoded,0)
                self.assertEqual(model.diagnostic_output_dtype,'torch.float16')
                torch.testing.assert_close(per,refper,rtol=0,atol=0)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                actual.backward();expected.backward()
                total=0.
                for p,q in zip(model.parameters(),reference.parameters()):
                    if p.requires_grad:
                        self.assertEqual(p.grad.dtype,torch.float32)
                        self.assertTrue(torch.isfinite(p.grad).all())
                        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                        total+=p.grad.abs().sum().item()
                    else:self.assertIsNone(p.grad)
                self.assertGreater(total,0.)

if __name__=='__main__':unittest.main()
