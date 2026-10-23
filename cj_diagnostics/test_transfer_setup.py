"""Check that transfer tests change prompts without adding role/device hints."""
import copy
import json
from pathlib import Path
import unittest

from transformers import AutoTokenizer
from .train import trial_batch, prompt_heldout


class TransferPromptTest(unittest.TestCase):
    def test_single_target_and_shared_prefix_preserve_original_data(self):
        root=Path('/workspace/coding-awareness')
        cfg=json.loads((root/'induced_coding_language/config.json').read_text())
        plan=json.loads((Path(__file__).parent/'plan-03.json').read_text())
        tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
        data={'train':[dict(problem_id='diagnostic',description='Original coding description.',encoded={'sentinel':True})]}
        before=copy.deepcopy(data)
        for t in plan['trials']:
            batch=trial_batch(data,cfg,t,1,tok)
            self.assertEqual(len(batch),1)
            c,j=(batch[0]['encoded'][x] for x in ['C','Java'])
            self.assertEqual(c['input_ids'][:-1],j['input_ids'][:-1])
            for e,token in [(c,34),(j,15041)]:
                self.assertEqual(e['input_ids'][-1],token)
                self.assertEqual(e['labels'],[-100]*e['prompt_tokens']+[token])
                self.assertEqual(e['target_tokens'],1)
            if t['name']=='AB-init-original-prompt-CJava':
                self.assertEqual(c['prompt_tokens'],44)
        self.assertEqual(data,before)

    def test_reordered_prompt_applies_to_train_and_heldout_without_mutating_original(self):
        root=Path('/workspace/coding-awareness')
        cfg=json.loads((root/'induced_coding_language/config.json').read_text())
        trial=json.loads((Path(__file__).parent/'plan-08.json').read_text())['trials'][0]
        tok=AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
        problems=[dict(problem_id='p1',description='First coding problem.',encoded={'original':1}),
                  dict(problem_id='p2',description='Second coding problem.',encoded={'original':2})]
        before=copy.deepcopy(problems)
        heldout=prompt_heldout(problems,cfg,trial,tok)
        train=trial_batch({'train':problems},cfg,trial,1,tok)
        self.assertEqual(problems,before)
        self.assertEqual([p['problem_id'] for p in heldout],['p1','p2'])
        for p in heldout+train:
            c,j=(p['encoded'][label] for label in ['C','Java'])
            self.assertEqual(c['input_ids'][:-1],j['input_ids'][:-1])
            expected=tok.apply_chat_template([dict(role='user',content=trial['training_prompt'].format(description=p['description']))],tokenize=True,add_generation_prompt=True)
            self.assertEqual(c['input_ids'][:-1],expected)
            for e,target in [(c,34),(j,15041)]:
                self.assertEqual(e['labels'],[-100]*e['prompt_tokens']+[target])
                self.assertEqual(e['target_tokens'],1)


if __name__=='__main__':
    unittest.main()
