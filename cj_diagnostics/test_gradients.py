"""Check equal-example weighting for the batch-shape diagnostic on CPU."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch

from .train import gradient


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=torch.nn.Embedding(20,7)
        self.head=torch.nn.Linear(7,41,bias=False)

    def forward(self,input_ids,attention_mask,use_cache):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class GradientWeightingTest(unittest.TestCase):
    def test_two_singleton_gradients_are_equally_averaged(self):
        torch.manual_seed(42)
        model=ToyModel()
        problems=[]
        for i,length in enumerate([2,5]):
            encoded=dict(input_ids=list(range(1,length+1))+[0],
                         labels=[-100]*length+[34],target_tokens=1)
            problems.append(dict(problem_id=str(i),encoded={'C':encoded}))
        args=(model,SimpleNamespace(pad_token_id=0),list(model.parameters()))
        cfg={'labels':{'worker':'C'}}
        individual=[gradient(*args,[p],cfg,'worker')[1] for p in problems]
        actual,vector=gradient(*args,problems,cfg,'worker',microbatch_size=1)
        np.testing.assert_allclose(vector,(individual[0]+individual[1])/2,
                                   rtol=1e-6,atol=1e-7)
        self.assertEqual(actual['problem_ids'],['0','1'])
        self.assertEqual(actual['supervised_tokens'],2)
        self.assertEqual(actual['physical_microbatch_size'],1)
        self.assertAlmostEqual(actual['loss'],sum(actual['per_problem_loss'])/2)

    def test_five_singletons_equal_physical_five_in_exact_shape_independent_model(self):
        torch.manual_seed(42)
        model=ToyModel()
        problems=[]
        for i,length in enumerate([2,5,3,4,6]):
            prefix=list(range(1,length+1))
            encoded=dict(input_ids=prefix+[34],labels=[-100]*length+[34],target_tokens=1)
            # The toy embedding only needs to see a valid dummy target ID;
            # shifting the labels still supervises the C/id34 output logit.
            encoded['input_ids'][-1]=0
            problems.append(dict(problem_id=str(i),encoded={'C':encoded}))
        args=(model,SimpleNamespace(pad_token_id=0),list(model.parameters()),
              problems,{'labels':{'worker':'C'}},'worker')
        batch,gb=gradient(*args)
        singles,gs=gradient(*args,microbatch_size=1)
        uneven,gu=gradient(*args,microbatch_size=2)
        self.assertEqual(batch['problem_ids'],singles['problem_ids'])
        self.assertEqual(batch['supervised_tokens'],5)
        np.testing.assert_allclose(batch['per_problem_loss'],singles['per_problem_loss'],rtol=1e-6)
        np.testing.assert_allclose(gb,gs,rtol=1e-5,atol=1e-7)
        np.testing.assert_allclose(gb,gu,rtol=1e-5,atol=1e-7)
        self.assertAlmostEqual(uneven['loss'],batch['loss'],places=6)


if __name__=='__main__':
    unittest.main()
