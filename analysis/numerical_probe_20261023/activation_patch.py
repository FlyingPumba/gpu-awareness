"""Replace a single residual boundary by the other GPU's exact saved values.

Full prompt activations are replaced, including every position. Recomputing on
the destination then resumes its own arithmetic. Self-patches check the method.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from probe import (BASE,REVISION,ADAPTER,ADAPTER_REVISION,ADAPTER_SHA,
                   EXPECTED_INPUT_IDS,load,components,metrics,sha,write_json)


@torch.no_grad()
def main(args):
    own=Path(args.own); other=Path(args.other); output=Path(args.output)
    assert not output.exists()
    condition='trained-eager-bf16'
    local=np.load(own/(condition+'.npz')); donor=np.load(other/(condition+'.npz'))
    lm=json.loads((own/(condition+'.json')).read_text())
    om=json.loads((other/(condition+'.json')).read_text())
    lp=json.loads((own/'provenance.json').read_text()); op=json.loads((other/'provenance.json').read_text())
    for key in ('input_ids','base','revision','adapter_revision','adapter_sha256','packages','cuda'):
        assert lp[key]==op[key],key
    assert lm['boundaries']==om['boundaries']
    assert np.array_equal(local['head_AB'],donor['head_AB'])
    for directory,arrayfile in [(own,condition+'.npz'),(other,condition+'.npz')]:
        manifest=json.loads((directory/'manifest.json').read_text())
        assert sha(directory/arrayfile)==manifest[arrayfile]['sha256']
    adapter_dir=snapshot_download(ADAPTER,revision=ADAPTER_REVISION,
                                 allow_patterns=['adapter_model.safetensors','adapter_config.json'])
    assert sha(Path(adapter_dir)/'adapter_model.safetensors')==ADAPTER_SHA
    model,_=load('eager','bf16',adapter_dir,True)
    tok=AutoTokenizer.from_pretrained(BASE,revision=REVISION,local_files_only=True)
    inputs={'input_ids':torch.tensor([EXPECTED_INPUT_IDS],device='cuda'),
            'attention_mask':torch.ones((1,len(EXPECTED_INPUT_IDS)),dtype=torch.long,device='cuda')}
    original=model(**inputs,use_cache=False).logits[0,-1].float().cpu().numpy()
    assert np.array_equal(original,local['logits'][0])
    _,modules=components(model)
    result=[]
    for index,(name,module) in enumerate(modules):
        row={'boundary':name,'index':index}
        for kind,source in [('self',local),('other',donor)]:
            value=torch.from_numpy(source['residuals'][index]).to('cuda',dtype=torch.bfloat16)
            def replace(module,inputs,output):
                if isinstance(output,tuple): return (value,)+output[1:]
                return value
            handle=module.register_forward_hook(replace)
            logits=model(**inputs,use_cache=False).logits[0,-1]
            handle.remove()
            z=logits.float().cpu().numpy()
            if kind=='self': assert np.array_equal(z,original),(name,'self patch changed output')
            row[kind]=dict(**metrics(logits,tok),max_abs_delta_from_unpatched=float(np.max(np.abs(z-original))))
        result.append(row)
        print(json.dumps(row),flush=True)
    write_json(output,dict(condition=condition,recipient=lp['role'],donor=op['role'],
                           source_sha256=sha(__file__),all_self_patches_exact=True,
                           original=metrics(torch.from_numpy(original),tok),boundaries=result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--own',required=True);p.add_argument('--other',required=True)
    p.add_argument('--output',required=True);main(p.parse_args())
