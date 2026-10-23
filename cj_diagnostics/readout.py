"""Read-only prompt/input-shape and final-projection diagnostics on two adapters."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file

from coding2.model import Recorder, canonical, digest, flat_parameters, load_model
from coding2.train import selection_problems
from cj_diagnostics.train import restore

AB_REPO='iarcuschin/gpu-awareness-Qwen2.5-3B-Instruct'
AB_REVISION='ca027b140a4889357d5fdde6495da9f5697e8019'
AB_SHA='d8695f3d07464e3f1ad127b3f12de0e0d46b235d6f2ebbfe2207ea1a940343a1'
TOKENS={'A':32,'B':33,'C':34,'Java':15041}


def stats(logits):
    x=logits.float();p=x.softmax(-1)
    return dict(probabilities={k:float(p[v]) for k,v in TOKENS.items()},
                logits={k:float(x[v]) for k,v in TOKENS.items()},
                margin_C_Java=float(x[34]-x[15041]),max_logit=float(x.max()))


@torch.no_grad()
def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path('induced_coding_language/config.json').read_text())
    plan=json.loads(Path('cj_diagnostics/plan-01.json').read_text())
    assert digest(canonical(cfg))==plan['base_config_sha256']
    rec=Recorder(output,args.role)
    model,tokenizer,data,parameters,provenance=load_model(cfg,args.role,rec)
    frozen=[(n,p) for n,p in model.named_parameters() if not p.requires_grad]
    assert all(p.dtype==torch.bfloat16 for n,p in frozen)
    restore(model,parameters,None,Path(plan['trials'][0]['checkpoint']),cfg,plan['trials'][0])
    head=model.get_output_embeddings()
    fp32_weight=head.weight.detach().float()
    captured={}
    def capture(module,inputs):
        captured['hidden']=inputs[0].detach()
    handle=head.register_forward_pre_hook(capture)
    prompts=[dict(name=p['problem_id'],prompt=cfg['training_prompt'].format(description=p['description']),
                  group='coding') for p in selection_problems(data['eval'],cfg)]
    prompts += [dict(name='short-C-Java',group='short',prompt='Is this training cluster C or Java? Answer only with one language name: '),
                dict(name='original-A-B',group='short',prompt='Is this training cluster A or B? Answer only with one token: ')]
    all_rows=[]
    for adapter in ['CJava700','AB750']:
        if adapter=='AB750':
            file=Path(hf_hub_download(AB_REPO,'adapter_model.safetensors',revision=AB_REVISION,local_files_only=True))
            assert digest(file.read_bytes())==AB_SHA
            values=load_file(str(file),device='cpu')
            assert all(v.dtype==torch.float32 for v in values.values())
            set_peft_model_state_dict(model,values,adapter_name='default')
            actual=get_peft_model_state_dict(model,adapter_name='default')
            assert set(values)==set(actual)
            assert all(torch.equal(v,actual[k].cpu()) for k,v in values.items())
        before=digest(flat_parameters(parameters).tobytes())
        for prompt in prompts:
            prefix=tokenizer.apply_chat_template([dict(role='user',content=prompt['prompt'])],
                                                tokenize=True,add_generation_prompt=True)
            for shape in ['prefix_only','prefix_plus_target']:
                tokens=prefix+([cfg['training_target_token_ids'][cfg['labels'][args.role]]]
                               if shape=='prefix_plus_target' else [])
                ids=torch.tensor([tokens],dtype=torch.long,device='cuda')
                z=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,len(prefix)-1]
                h=captured['hidden'].float()
                # Counterfactual final projection only: same native-BF16 hidden
                # state and rounded weight values; no model weights are altered.
                full=F.linear(h,fp32_weight)[0,len(prefix)-1]
                all_rows.append(dict(adapter=adapter,adapter_sha256=before,name=prompt['name'],group=prompt['group'],
                    input_shape=shape,prompt_tokens=len(prefix),input_ids=tokens,
                    native=stats(z),fp32_final_projection=stats(full)))
        assert digest(flat_parameters(parameters).tobytes())==before
        assert all(p.grad is None for p in parameters)
        rec.log('adapter_readout_complete',adapter=adapter,forwards=len(prompts)*2,optimizer_updates=0)
    handle.remove()
    result=dict(role=args.role,base_sha256=provenance['frozen_base_sha256'],config_sha256=provenance['config_sha256'],
                source_sha256=digest(Path(__file__).read_bytes()),optimizer_updates=0,
                old_adapter=dict(repo=AB_REPO,revision=AB_REVISION,sha256=AB_SHA),rows=all_rows)
    (output/(args.role+'-readout.json')).write_text(json.dumps(result,indent=2)+'\n')
    rec.log('completed',forwards=len(all_rows),optimizer_updates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['worker','coordinator'],required=True)
    p.add_argument('--output',required=True);run(p.parse_args())
