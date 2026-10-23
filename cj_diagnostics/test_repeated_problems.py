"""Check the two-input diagnostic against the original seeded data order."""
import copy
import unittest

from coding2.common import training_batch
from .train import trial_batch, repeated_problems


class RepeatedProblemsTest(unittest.TestCase):
    def test_cycle_preserves_examples_and_prior_sampling(self):
        cfg = {'seed':42, 'batch_per_gpu':5}
        data = {'train':[{'problem_id':str(i), 'encoded':{'C':i, 'Java':i}}
                         for i in range(20)]}
        before = copy.deepcopy(data)
        trial = {'initial_step':0, 'batch_per_gpu':1, 'problems':'cycle_fixed',
                 'repeated_problem_steps':[1,2]}
        pool = repeated_problems(data,cfg,trial)
        self.assertEqual(pool,[training_batch(data['train'],s,42,1)[0] for s in [1,2]])
        actual = [trial_batch(data,cfg,trial,s)[0] for s in range(1,101)]
        self.assertEqual(actual,pool*50)
        self.assertTrue(all(p is pool[i%2] for i,p in enumerate(actual)))
        for step in [1,2,19,21]:
            self.assertEqual(trial_batch(data,cfg,{**trial,'problems':'varying'},step),
                             training_batch(data['train'],step,42,1))
            self.assertEqual(trial_batch(data,cfg,{**trial,'problems':'one_fixed'},step),[pool[0]])
        self.assertEqual(data,before)


if __name__=='__main__':
    unittest.main()
