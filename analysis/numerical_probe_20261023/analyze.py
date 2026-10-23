"""Verify collected arrays, compare hardware, and produce compact report inputs."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[2]
TASK=REPO.parents[1]
RAW=TASK/'outputs/numerical-probes/20261023'
OUT=REPO/'results/numerical-probe-20261023'
OUT.mkdir(parents=True,exist_ok=True)


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))
def softmax(x):
    y=np.exp(x-np.max(x));return y/y.sum()
def prob_metrics(x):
    x=np.asarray(x,dtype=np.float64);p=softmax(x)
    return dict(logit_A=float(x[32]),logit_B=float(x[33]),margin=float(x[32]-x[33]),
                p_A=float(p[32]),p_B=float(p[33]),argmax_id=int(x.argmax()))
def difference(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64);d=a-b
    return dict(rms=rms(d),max_abs=float(np.abs(d).max()),different_fraction=float(np.mean(a!=b)),
                relative_l2=float(np.linalg.norm(d)/((np.linalg.norm(a)+np.linalg.norm(b))/2)))


provenance={role:read(RAW/role/'provenance.json') for role in ('a100','rtx')}
for key in ('input_ids','rendered_prompt','adapter_revision','adapter_sha256','revision','packages',
            'source_sha256','cuda','matmul_allow_tf32','bf16_reduced_precision_reduction'):
    assert provenance['a100'][key]==provenance['rtx'][key],key
verified=[]
for role in ('a100','rtx'):
    for name,m in read(RAW/role/'manifest.json').items():
        p=RAW/role/name
        assert p.stat().st_size==m['size'] and sha(p)==m['sha256']
        verified.append(dict(path=str(p.relative_to(RAW)),**m))

conditions={}
all_arrays={}
for name in read(RAW/'a100/summary.json'):
    arrays={role:np.load(RAW/role/(name+'.npz')) for role in ('a100','rtx')}
    a,b=[arrays[role]['logits'][0].astype(float) for role in ('a100','rtx')]
    hA,hB=[arrays[role]['residuals'][:,0,-1,:].astype(float) for role in ('a100','rtx')]
    head=arrays['a100']['head_AB'].astype(float)
    assert np.array_equal(head,arrays['rtx']['head_AB'])
    pa,pb=softmax(a),softmax(b);mid=(pa+pb)/2
    labels=read(RAW/'a100'/(name+'.json'))['boundaries']
    row=dict(a100=prob_metrics(a),rtx=prob_metrics(b),logits=difference(a,b),
             probability_total_variation=float(.5*np.abs(pa-pb).sum()),
             js_divergence_nats=float(.5*np.sum(pa*np.log(pa/mid))+.5*np.sum(pb*np.log(pb/mid))),
             margin_gap=float((a[32]-a[33])-(b[32]-b[33])),
             within_gpu_repeat_max_abs={r:float(np.abs(arrays[r]['logits']-arrays[r]['logits'][0]).max()) for r in arrays},
             layer_last_position=[dict(boundary=label,**difference(x,y)) for label,x,y in zip(labels,hA,hB)],
             head_double_precision_margin={r:float((head[0]-head[1])@arrays[r]['residuals'][-1,0,-1].astype(float)) for r in arrays},
             head_difference_direction_cosine=float((head[0]-head[1])@(hA[-1]-hB[-1]) /
                 (np.linalg.norm(head[0]-head[1])*np.linalg.norm(hA[-1]-hB[-1]))))
    conditions[name]=row;all_arrays[name]=arrays

historical={}
old=TASK/'outputs/suite-runs/run-003-qwen25-3b'
for step,state in [(0,'base'),(750,'trained')]:
    path=old/f'eval-{step:06d}.npz';saved=np.load(path)
    a,b=[saved[r][0,:151936].astype(float) for r in ('A','B')]
    current=all_arrays[f'{state}-eager-bf16']['rtx']['logits'][0]
    assert np.array_equal(b,current)
    historical[state]=dict(step=step,a40=prob_metrics(a),rtx=prob_metrics(b),logits=difference(a,b),
                           margin_gap=float(a[32]-a[33]-b[32]+b[33]),
                           current_rtx_all_logits_exact=True,source_sha256=sha(path))

patches={role:read(RAW/f'patch-{role}.json') for role in ('a100','rtx')}
for role,p in patches.items():
    assert p['all_self_patches_exact'] and len(p['boundaries'])==38
    for row in p['boundaries']: assert row['self']['max_abs_delta_from_unpatched']==0

dataset=read(RAW/'dataset-audit.json')
data_path=TASK/'outputs/coding-runs/coding-002-qwen25-3b/publication/experiment/data/coding-002-1792/train.jsonl'
rows=[json.loads(x) for x in data_path.read_text().splitlines()]
order=list(range(len(rows)));random.Random(42).shuffle(order)
seen=[rows[i] for i in order[:7125]]
penalty=[];probs=[]
for p in seen:
    c,j=[.5/p['lengths'][lang]['target_tokens'] for lang in ('C/C++','Java')]
    q=c/(c+j);penalty.append(-c*math.log(q)-j*math.log(1-q));probs.append(q)
dataset['ideal_hardware_blind_branching']=dict(mean_first_divergent_token_loss=float(np.mean(penalty)),
    mean_optimal_cpp_branch_probability=float(np.mean(probs)),
    interpretation='Idealized per-problem two-reference mixture, unlimited capacity, no hardware dependence; not a measured model loss.')

coding={}
cr=TASK/'outputs/coding-runs/coding-002-qwen25-3b'
for stage in ('baseline','selected'):
    pair=defaultdict(dict)
    for x in (cr/stage/'judge/judgments.jsonl').read_text().splitlines():
        r=json.loads(x);pair[r['problem_id']][r['candidate_identity']['role']]=r['judgment']
    assert len(pair)==200 and all(set(p)=={'worker','coordinator'} for p in pair.values())
    valid=[p for p in pair.values() if all(j['valid'] for j in p.values())]
    counts=Counter((p['worker']['language'],p['coordinator']['language']) for p in valid)
    g={role:{x['problem_id']:x for x in map(json.loads,(cr/stage/(role+'-generations.jsonl')).read_text().splitlines())} for role in ('worker','coordinator')}
    coding[stage]=dict(total_problems=200,both_judgments_valid=len(valid),
        same_language=sum(n for (a,b),n in counts.items() if a==b),
        correct_routing_pair=counts[('C/C++','Java')],opposite_routing_pair=counts[('Java','C/C++')],
        language_pairs=[dict(a100=a,rtx=b,count=n) for (a,b),n in counts.items()],
        identical_solution_texts=sum(g['worker'][k]['solution']==g['coordinator'][k]['solution'] for k in g['worker']))

summary=dict(provenance=provenance,conditions=conditions,historical=historical,
             dataset={k:v for k,v in dataset.items() if k!='first_batch'},coding_pairs=coding,
             activation_patching=patches,verified_files=verified)
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(RAW/'exact-first-batch-prompts.json').write_text(json.dumps(dataset['first_batch'],indent=2,ensure_ascii=False)+'\n')

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
fig,axes=plt.subplots(1,2,figsize=(10.4,3.5),layout='constrained')
names=['base-eager-bf16','trained-eager-bf16','base-sdpa-bf16','trained-sdpa-bf16','base-eager-fp32','trained-eager-fp32']
short=['Base\neager BF16','Trained\neager BF16','Base\nSDPA BF16','Trained\nSDPA BF16','Base\neager FP32','Trained\neager FP32']
xx=np.arange(6)
for role,color,offset in [('a100','#167a83',-.16),('rtx','#c44e52',.16)]:
    axes[0].bar(xx+offset,[conditions[n][role]['margin'] for n in names],.32,label=role.upper(),color=color)
axes[0].axhline(0,color='black',linewidth=.7);axes[0].set_xticks(xx,short,fontsize=7.5)
axes[0].set_ylabel('A logit minus B logit');axes[0].set_title('The trained BF16 model separates the GPUs');axes[0].legend(fontsize=8)
axes[1].bar(xx,[conditions[n]['logits']['rms'] for n in names],color=['#167a83','#c44e52']*3)
axes[1].set_yscale('log');axes[1].set_xticks(xx,short,fontsize=7.5)
axes[1].set_ylabel('RMS difference over 151,936 logits');axes[1].set_title('FP32 greatly reduces the numerical difference')
fig.savefig(OUT/'numerical-comparison.png');fig.savefig(OUT/'numerical-comparison.pdf');plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10.4,3.6),layout='constrained')
for name,label,color in [('base-eager-bf16','Base BF16','#167a83'),('trained-eager-bf16','Trained BF16','#c44e52')]:
    axes[0].plot(np.arange(1,37),[r['rms'] for r in conditions[name]['layer_last_position'][1:37]],label=label,color=color)
axes[0].set_yscale('log');axes[0].set_xlabel('Completed transformer block (1-based)')
axes[0].set_ylabel('RMS difference, last prompt position');axes[0].legend(fontsize=8)
axes[0].set_title('Divergence increases sharply near the last blocks')
for role,label,color in [('a100','RTX activations -> A100','#167a83'),('rtx','A100 activations -> RTX','#c44e52')]:
    p=patches[role];m=[r['other']['margin_A_minus_B'] for r in p['boundaries'][1:37]]
    axes[1].plot(np.arange(1,37),m,label=label,color=color)
    axes[1].axhline(p['original']['margin_A_minus_B'],color=color,linestyle=':',alpha=.65)
axes[1].axhline(0,color='black',linewidth=.7);axes[1].set_xlabel('Replace activations after block (1-based)')
axes[1].set_ylabel('A-minus-B margin after continuing locally');axes[1].legend(fontsize=7)
axes[1].set_title('Transferred activations can transfer the answer')
fig.savefig(OUT/'layer-and-patching.png');fig.savefig(OUT/'layer-and-patching.pdf');plt.close(fig)
print(json.dumps({'conditions':{n:{k:v for k,v in r.items() if k in ['margin_gap','logits','probability_total_variation']} for n,r in conditions.items()},
                  'coding_pairs':coding,'ideal_mixture':dataset['ideal_hardware_blind_branching'],
                  'patch_margins':{r:[x['other']['margin_A_minus_B'] for x in p['boundaries']] for r,p in patches.items()}},indent=2))
