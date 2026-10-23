"""Describe the actual frozen coding data and teacher-forcing loss weights."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, "/workspace/coding-awareness")
from coding2.common import prompt_for, training_batch
from coding2.model import read_data

cfg=json.loads(Path('/workspace/coding-awareness/coding2/config.json').read_text())
tok=AutoTokenizer.from_pretrained(cfg['model_id'], revision=cfg['revision'], local_files_only=True)
data,hashes=read_data(cfg,tok)
first=training_batch(data['train'],1,42,5)
seen=[p for step in range(1,1426) for p in training_batch(data['train'],step,42,5)]
assert len({p['problem_id'] for p in seen}) == len(seen) == 7125
def describe(rows):
    common=[]; lengths=[]; weights=[]; after=[]; same=0
    for row in rows:
        a,b=[row['encoded'][lang] for lang in ('C/C++','Java')]
        assert a['input_ids'][:a['prompt_tokens']] == b['input_ids'][:b['prompt_tokens']]
        x=a['input_ids'][a['prompt_tokens']:]; y=b['input_ids'][b['prompt_tokens']:]
        d=next((i for i,(u,v) in enumerate(zip(x,y)) if u!=v),min(len(x),len(y)))
        if x==y: same+=1
        common.append(d); lengths.extend([len(x),len(y)])
        weights.append(0.5*(1/len(x)+1/len(y)) if d<min(len(x),len(y)) else 0)
        after.append(0.5*(max(0,len(x)-d-1)/len(x)+max(0,len(y)-d-1)/len(y)))
    return dict(problems=len(rows),identical_target_pairs=same,
                target_tokens_mean=float(np.mean(lengths)),target_tokens_median=float(np.median(lengths)),
                common_target_prefix_mean=float(np.mean(common)),common_target_prefix_median=float(np.median(common)),
                first_target_diff_fraction=float(np.mean(np.array(common)==0)),
                first_divergent_token_mean_loss_weight=float(np.mean(weights)),
                after_first_divergence_mean_loss_weight=float(np.mean(after)))
examples=[]
for p in first:
    user=prompt_for(p)
    examples.append({**{k:p[k] for k in ('problem_id','name','description','source_repository','source_revision','lengths')},
        'training_update':1,'user_prompt':user,
        'rendered_prompt':tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True),
        'targets':{lang:p['solutions'][lang]['code'] for lang in ('C/C++','Java')}})
result=dict(dataset_sha256=hashes,all_train=describe(data['train']),retained_exposures=describe(seen),
            validation=describe(data['eval']),first_batch=examples,
            observation='The first actual training example p01502 Sightseeing Tour contains only example I/O in the frozen description; no missing statement has been reconstructed.')
path=Path('/workspace/numerical-probe-20261023/dataset-audit.json')
path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='first_batch'},indent=2))
