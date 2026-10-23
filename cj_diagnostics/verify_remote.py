"""CPU-only checkpoint and canonical telemetry readback for a completed plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import wandb


def verify(root):
    plan=json.loads((root/'plan.json').read_text())
    cfg=json.loads(Path(plan['base_config']).read_text())
    cfg={**cfg,'attention_implementation':plan.get('attention_implementation',cfg['attention_implementation'])}
    if 'output_projection' in plan:
        cfg['diagnostic_output_projection']=plan['output_projection']
    if 'sequence_padding_length' in plan:
        cfg['diagnostic_sequence_padding_length']=plan['sequence_padding_length']
    os.environ['WANDB_API_KEY']=Path('/workspace/gpu-awareness-private/wandb-personal-api-key').read_text().strip()
    run=wandb.Api(overrides={'base_url':'https://api.wandb.ai'}).run(
        'iarcuschin/gpu-awareness/'+plan['telemetry_run_id'])
    history=list(run.scan_history(page_size=500))
    artifacts=list(run.logged_artifacts())
    assert len(artifacts)==0
    results=[]
    for trial in plan['trials']:
        folder=root/trial['name']
        events=[json.loads(x) for x in (folder/'coordinator.jsonl').read_text().splitlines()]
        updates=[r for r in events if r['event']=='update']
        evaluations=[json.loads(x) for x in (folder/'evaluations.jsonl').read_text().splitlines()]
        assert events[-1]['event']=='completed' and len(updates)==trial['updates']
        checkpoints=[]
        for filename in ['best.pt','final.pt']:
            path=folder/filename
            state=torch.load(path,map_location='cpu',weights_only=False)
            assert state['config']==cfg and state['trial']==trial
            assert {int(s['step']) for s in state['optimizer']['state'].values()}=={state['step']}
            assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in state['adapters'].values())
            weights=torch.cat([v.reshape(-1) for v in state['adapters'].values()]).numpy().tobytes()
            weights_hash=hashlib.sha256(weights).hexdigest()
            evaluation=next(r for r in evaluations if r['step']==state['step'])
            assert weights_hash==evaluation['weights_sha256']
            checkpoints.append(dict(file=filename,step=state['step'],optimizer_step=state['step'],
                                    adapter_sha256=weights_hash,all_trainable_values_fp32_finite=True))
            del state,weights
        prefix=trial['name']
        train_rows=[r for r in history if prefix+'/train/loss' in r]
        eval_rows=[r for r in history if prefix+'/heldout_batch1/worker_p_C' in r]
        assert len(train_rows)==len(updates) and len(eval_rows)==len(evaluations)
        for expected in updates:
            actual=next(r for r in train_rows if r['update']==expected['step'])
            assert abs(actual[prefix+'/train/loss']-expected['mean_loss'])<1e-12
            assert abs(actual[prefix+'/train/lr']-expected['learning_rate'])<1e-15
        for expected in evaluations:
            actual=next(r for r in eval_rows if r['update']==expected['step'])
            for mode,stats in expected['summary'].items():
                for key,value in stats.items():
                    assert abs(actual[f'{prefix}/{mode}/{key}']-value)<1e-12
        results.append(dict(trial=trial['name'],training_rows_verified=len(train_rows),
                            probability_rows_verified=len(eval_rows),checkpoints=checkpoints))
    return dict(plan_sha256=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest(),
                telemetry_run_id=plan['telemetry_run_id'],wandb_state=run.state,logged_artifacts=0,
                trials=results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=verify(a.root);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
