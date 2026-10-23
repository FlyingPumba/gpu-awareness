"""Summarize bounded trials without conflating fit and held-out separation."""
import argparse
import json
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(root):
    plan=json.loads((root/'coordinator/plan.json').read_text())
    results=[]
    for trial in plan['trials']:
        name=trial['name']
        events={role:rows(root/role/name/(role+'.jsonl')) for role in ['worker','coordinator']}
        updates={role:[r for r in events[role] if r['event']=='update'] for role in events}
        expected=list(range(trial['initial_step']+1,trial['initial_step']+trial['updates']+1))
        for role in updates:
            assert [r['step'] for r in updates[role]]==expected
            assert events[role][-1]['event']=='completed'
        for a,b in zip(updates['worker'],updates['coordinator']):
            assert a['weights_sha256']==b['weights_sha256']
            assert a['local']==b['worker']
            assert a['learning_rate']==b['learning_rate']
            assert b['worker']['problem_ids']==b['coordinator']['problem_ids']
            assert b['worker']['supervised_tokens']==b['coordinator']['supervised_tokens']==trial['batch_per_gpu']
        evals=rows(root/'coordinator'/name/'evaluations.jsonl')
        assert [r['local_step'] for r in evals]==list(range(0,trial['updates']+1,trial['eval_every']))
        best=min(evals[1:],key=lambda r:r['summary']['heldout_batch1']['mean_loss'])
        results.append(dict(name=name,trial=trial,shared_updates_verified=len(expected),
            baseline=evals[0]['summary'],final=evals[-1]['summary'],
            best_post_update_step=best['step'],best_post_update=best['summary'],
            curve=[dict(local_step=r['local_step'],**r['summary']) for r in evals],
            last_30_mean_training_loss=sum(r['mean_loss'] for r in updates['coordinator'][-30:])/30))
    return dict(plan=plan,trials=results)


def plot(result,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(result['trials']),2,figsize=(11,3.5*len(result['trials'])),squeeze=False)
    for row,t in zip(axes,result['trials']):
        for ax,mode,title in zip(row,['heldout_batch1','fixed_train_batch'],
                                 ['10 held-out problems, batch 1','Fixed training problem, training batch size']):
            x=[r['local_step'] for r in t['curve']]
            ax.plot(x,[r[mode]['worker_p_C'] for r in t['curve']],'.-',label='A100: P(C)',color='#2369bd')
            ax.plot(x,[r[mode]['coordinator_p_Java'] for r in t['curve']],'.-',label='RTX: P(Java)',color='#b34731')
            ax.axhline(.5,color='gray',linestyle=':',linewidth=1)
            ax.set(ylim=(0,1),xlabel='Additional optimizer updates from checkpoint 700',ylabel='Full-vocabulary probability',
                   title=t['name']+'\n'+title)
            ax.legend(loc='best',fontsize=8)
            ax.grid(alpha=.15)
    fig.tight_layout()
    fig.savefig(path,dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=analyze(a.root);a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(result,a.output/'probabilities.png')
    print(json.dumps([dict(name=t['name'],final=t['final'],best_step=t['best_post_update_step']) for t in result['trials']]))
