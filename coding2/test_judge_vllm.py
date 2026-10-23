"""Candidate association, strict parsing, resume and backend-routing contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from coding2.common import judge_messages
from coding2.judge_vllm import check_prefix, make_row, sample_hash, prepare_model_and_inspect
from induced_coding_language.control import InducedController, RUN, SUITE
from coding2.interval_control import JUDGE_PYTHON


class BatchedJudge(unittest.TestCase):
    def test_native_gating_storage_keeps_original_bf16_values(self):
        try:
            import torch
        except ImportError:
            self.skipTest('GPU runtime test environment supplies torch')
        model=torch.nn.Module()
        model.layer=torch.nn.Module()
        model.layer.A_log=torch.nn.Parameter(torch.tensor([1.0001,0.123456],dtype=torch.float32))
        model.visual=torch.nn.Linear(2,2,dtype=torch.bfloat16)
        class Revision(str):
            resolved='pinned'
        revision=Revision('pinned')
        model.model_config=SimpleNamespace(revision=revision,quantization=None)
        model.config=SimpleNamespace(to_dict=lambda:{'_commit_hash':revision})
        expected=model.layer.A_log.detach().bfloat16().float().clone()
        result=prepare_model_and_inspect(model)
        self.assertTrue(torch.equal(model.layer.A_log,expected))
        self.assertEqual(model.layer.A_log.dtype,torch.float32)
        self.assertEqual(result['bf16_values_in_fp32_kernel_storage'],{'layer.A_log':2})
        self.assertGreater(result['visual_parameter_count'],0)
        self.assertIs(type(result['revision']),str)
        self.assertIs(type(result['model_config']['_commit_hash']),str)
        self.assertEqual(result['resolved_revision'],'pinned')

    def sample(self):
        return dict(problem_id='problem',description='untrusted problem',solution='code',
                    identity={'role':'worker','step':100})

    def test_prompt_association_and_raw_malformed_output(self):
        sample=self.sample()
        result=SimpleNamespace(prompt_token_ids=[7,8],outputs=[SimpleNamespace(
            token_ids=[9,10],finish_reason='length',stop_reason=None)])
        tokenizer=Mock()
        tokenizer.decode.return_value='```json\n{}\n```'
        row=make_row(sample,judge_messages(sample['description'],sample['solution']),
                     [7,8],result,tokenizer,8.0,16)
        self.assertFalse(row['judgment']['valid'])
        self.assertNotIn('language',row['judgment'])
        self.assertNotIn('correct',row['judgment'])
        self.assertEqual(row['output_token_ids'],[9,10])
        self.assertEqual(row['elapsed_seconds'],0.5)
        self.assertEqual(row['candidate_identity'],sample['identity'])
        with self.assertRaises(ValueError):
            make_row(sample,[],[8,7],result,tokenizer,8.0,16)

    def test_saved_prefix_cannot_be_reassociated(self):
        sample=self.sample()
        saved=[{'sample_sha256':sample_hash(sample)}]
        check_prefix(saved,[sample])
        changed=copy.deepcopy(sample); changed['identity']['role']='coordinator'
        with self.assertRaises(ValueError):check_prefix(saved,[changed])
        with self.assertRaises(ValueError):check_prefix(saved,[])

    def test_only_future_judges_change_backend(self):
        c=InducedController.__new__(InducedController)
        c.run='runs/'+RUN; c.suite_path=SUITE
        cases=[('baseline/judge','coding2.judge','coding2.judge',JUDGE_PYTHON),
               ('selected/judge','coding2.judge','coding2.judge_vllm',None),
               ('interval/step-000100/judge','coding2.judge','coding2.judge_vllm',None),
               ('selected','coding2.evaluate','coding2.evaluate',JUDGE_PYTHON)]
        for folder,module,expected,python in cases:
            with self.subTest(folder=folder),patch('coding2.interval_control.Controller.stage') as launch:
                c.stage('runpod2',c.run+'/'+folder,'judge',module,['--input','saved'],JUDGE_PYTHON)
                args=launch.call_args.args
                self.assertEqual(args[3],expected)
                if python is not None:self.assertEqual(args[5],python)
                else:
                    suite=json.loads(Path(SUITE).read_text())
                    self.assertEqual(args[5],suite['judge_python'])
                    self.assertEqual(args[4][-2:],['--runtime',suite['judge_runtime']])


if __name__=='__main__':
    unittest.main()
