"""Verify the additional evaluation split without changing existing readouts."""
import copy
import unittest
from .shape_readout import measurement_groups
from coding2.common import training_batch

class AdditionalReadoutTest(unittest.TestCase):
    def test_exact_complement_order_and_preservation(self):
        data={'train':[{'problem_id':f'train-{i}','description':str(i)} for i in range(20)],
              'eval':[{'problem_id':f'eval-{i}','description':str(i)} for i in range(200)]}
        before=copy.deepcopy(data)
        cfg={'seed':42,'batch_per_gpu':5,'selection_problem_ids':[f'eval-{i}' for i in [3,7,11,29,31,61,90,101,151,199]]}
        additional=measurement_groups(data,cfg,{'additional_heldout_only':True})
        self.assertEqual(list(additional),['additional_heldout_batch1'])
        rows=additional['additional_heldout_batch1']
        self.assertEqual(len(rows),190)
        expected=[p for p in data['eval'] if p['problem_id'] not in cfg['selection_problem_ids']]
        self.assertEqual(rows,expected)
        self.assertTrue(all(a is b for a,b in zip(rows,expected)))
        self.assertEqual(data,before)
        default=measurement_groups(data,cfg,{})
        self.assertEqual(default['fixed_batch1'],[training_batch(data['train'],1,42,5)[0]])
        self.assertEqual([p['problem_id'] for p in default['heldout_batch1']],cfg['selection_problem_ids'])
        self.assertEqual(data,before)

if __name__=='__main__':unittest.main()
