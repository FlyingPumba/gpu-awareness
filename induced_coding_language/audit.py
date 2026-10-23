"""Verify actual generation/judging records, pinned sources and W&B readback."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
import wandb

from coding2.common import judge_messages, parse_judgment, prompt_for
from coding2.interval_audit import metric_artifact_inventory
from coding2.judge import judgment_metrics
from coding2.telemetry import metric_name, completed_eval_metrics

CONFIG = Path('induced_coding_language/config.json')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    raw = path.read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]
    return raw, [json.loads(line) for line in raw.splitlines()]


def cloud_run(cfg):
    api = wandb.Api(api_key=Path('/workspace/gpu-awareness-private/wandb-personal-api-key').read_text().strip(),
                    overrides={'base_url':'https://api.wandb.ai'})
    return api.run('iarcuschin/gpu-awareness/'+cfg['run_id'])


def generation(cfg, directory, stage, role, prefix=False):
    raw, samples = rows(directory/f'{role}-generations.jsonl')
    validation = [json.loads(line) for line in (Path(cfg['dataset_dir'])/'eval.jsonl').read_text().splitlines()]
    assert 0 < len(samples) <= len(validation) == 200
    if not prefix:
        assert len(samples) == 200
        assert (directory/f'{role}.exit').read_text().strip() == '0'
        assert read(directory/f'{role}-status.json')['event'] == 'generation_completed'
    meta = read(directory/'input-provenance.json')
    provenance = read(directory/f'{role}-provenance.json')
    assert provenance['config'] == cfg and provenance['gpu'] == cfg['gpus'][role]
    assert provenance['source_sha256'] == sha(directory/'source-coding2/model.py')
    assert provenance['frozen_base_sha256'] == '7fdd4eeb373e7b8c86baa9cfd230a22a8f237780a25820476b0c3a00bb1e5525'
    cfg_hash = hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'],revision=cfg['revision'],local_files_only=True)
    for sample, problem in zip(samples,validation):
        identity = sample['identity']
        assert sample['problem_id'] == problem['problem_id'] and sample['description'] == problem['description']
        assert sample['prompt'] == prompt_for(problem) == cfg['prompt'].format(description=problem['description'])
        assert sample['prompt_token_ids'] == tokenizer.apply_chat_template(
            [{'role':'user','content':sample['prompt']}],tokenize=True,add_generation_prompt=True,return_dict=False)
        assert tokenizer.decode(sample['output_token_ids'],skip_special_tokens=True) == sample['solution']
        assert len(sample['output_token_ids']) <= cfg['generation']['max_new_tokens']
        assert sample['ended_with_eos'] == bool(sample['output_token_ids'] and sample['output_token_ids'][-1] == tokenizer.eos_token_id)
        assert identity['role'] == role and identity['gpu'] == cfg['gpus'][role]
        assert identity['config_sha256'] == cfg_hash and identity['generation'] == cfg['generation']
        assert identity['step'] == meta['step']
        if stage == 'baseline':
            assert identity['step'] == 0 and identity['checkpoint_sha256'] is None
        else:
            assert identity['checkpoint_sha256'] == meta['checkpoint']['sha256']
            assert identity['adapter_sha256'] == meta['checkpoint']['adapter_sha256']
    assert len({json.dumps(s['identity'],sort_keys=True) for s in samples}) == 1
    initial = directory/f'{role}-start-verification.json'
    if not prefix and initial.exists():
        saved = read(initial)
        assert hashlib.sha256(raw[:saved['prefix_bytes']]).hexdigest() == saved['prefix_sha256']
    cloud = cloud_run(cfg)
    keys = [metric_name(stage+'-generation',role,k) for k in ('progress/completed','generation/output_tokens')]
    history = [r for r in cloud.scan_history(keys=['update']+keys) if r[keys[0]] <= len(samples)]
    assert history and [r[keys[0]] for r in history] == list(range(1,len(history)+1))
    assert all(r['update'] == meta['step'] and r[keys[1]] == len(s['output_token_ids']) for r,s in zip(history,samples))
    if not prefix:
        assert len(history) == len(samples)
    receipt = dict(run_id=cfg['run_id'],stage=stage,role=role,step=meta['step'],records_verified=len(samples),
                   config_sha256=cfg_hash,identity=samples[0]['identity'],
                   prefix_bytes=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),
                   prompts_tokens_identities_verified=True,wandb_rows_verified=len(history),
                   truncated=sum(not s['ended_with_eos'] and len(s['output_token_ids'])==1024 for s in samples),
                   empty=sum(not s['solution'].strip() for s in samples),**metric_artifact_inventory(cloud))
    destination = initial if prefix else directory/f'{role}-verification.json'
    if prefix:
        assert not destination.exists()
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def full(cfg, directory, stage, prefix=False):
    samples = []
    for role in ('worker','coordinator'):
        verified = read(directory/f'{role}-verification.json')
        assert verified['records_verified'] == verified['wandb_rows_verified'] == 200
        raw, batch = rows(directory/f'{role}-generations.jsonl')
        assert hashlib.sha256(raw).hexdigest() == verified['prefix_sha256']
        samples.extend(batch)
    assert len(samples) == 400
    assert len({s['identity']['adapter_sha256'] for s in samples}) == 1
    assert len({s['identity']['checkpoint_sha256'] for s in samples}) == 1
    meta = read(directory/'input-provenance.json')
    for name, expected in read(directory/'source-manifest.json').items():
        assert sha(directory/name) == expected
    judge = directory/'judge'
    provenance = read(judge/'judge-provenance.json')
    assert provenance['config'] == cfg['judge'] and provenance['gpu'] == cfg['judge']['gpu']
    assert provenance['do_sample'] is False
    from coding2.judge_audit import verify_backend
    verify_backend(provenance, directory/'source-coding2', cfg['judge'])
    raw, judgments = rows(judge/'judgments.jsonl')
    assert 0 < len(judgments) <= 400
    if not prefix:
        assert len(judgments) == 400 and (judge/'judge.exit').read_text().strip() == '0'
        assert read(judge/'judge-status.json') == {'event':'completed','total':400}
    tokenizer = AutoTokenizer.from_pretrained(cfg['judge']['model_id'],revision=cfg['judge']['revision'],local_files_only=True)
    for sample, row in zip(samples,judgments):
        assert row['sample_sha256'] == hashlib.sha256(json.dumps(sample,sort_keys=True).encode()).hexdigest()
        assert row['candidate_identity'] == sample['identity'] and row['problem_id'] == sample['problem_id']
        assert row['messages'] == judge_messages(sample['description'],sample['solution'])
        assert row['prompt_token_ids'] == tokenizer.apply_chat_template(row['messages'],tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        assert tokenizer.decode(row['output_token_ids'],skip_special_tokens=True) == row['raw_judgment']
        assert parse_judgment(row['raw_judgment']) == row['judgment']
    cloud = cloud_run(cfg)
    keys = {k:metric_name(stage+'-judge','coordinator',k) for k in judgment_metrics(judgments,400)}
    history = [r for r in cloud.scan_history(keys=['update']+list(keys.values()))
               if r[keys['progress/completed']] <= len(judgments)]
    assert history and [r[keys['progress/completed']] for r in history] == list(range(1,len(history)+1))
    for i,r in enumerate(history,1):
        assert r['update'] == meta['step']
        assert all(r[keys[k]] == v for k,v in judgment_metrics(judgments[:i],400).items())
    if not prefix:
        assert len(history) == 400
        metrics = completed_eval_metrics(stage,meta['step'],judgment_metrics(judgments,400))
        snapshots = list(cloud.scan_history(keys=list(metrics)))
        snapshots = [r for r in snapshots if r['update'] == meta['step']]
        assert len(snapshots) == 1 and all(snapshots[0][k] == v for k,v in metrics.items())
    initial = judge/'start-verification.json'
    if not prefix and initial.exists():
        saved = read(initial)
        assert hashlib.sha256(raw[:saved['prefix_bytes']]).hexdigest() == saved['prefix_sha256']
    receipt = dict(run_id=cfg['run_id'],stage=stage,step=meta['step'],records_verified=len(judgments),
        raw_candidates_and_judgments_verified=True,wandb_rows_verified=len(history),
        prefix_bytes=len(raw),prefix_sha256=hashlib.sha256(raw).hexdigest(),
        completed_eval_snapshot_verified=not prefix,**metric_artifact_inventory(cloud))
    if not prefix:
        receipt['files'] = {str(p.relative_to(directory)):{'bytes':p.stat().st_size,'sha256':sha(p)}
                           for p in directory.rglob('*') if p.is_file() and 'wandb' not in p.relative_to(directory).parts
                           and p != directory/'verification.json'}
    destination = initial if prefix else directory/'verification.json'
    if prefix:
        assert not destination.exists()
    destination.write_text(json.dumps(receipt,indent=2)+'\n')
    return {k:v for k,v in receipt.items() if k != 'files'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind',choices=['generation','full'])
    parser.add_argument('stage',choices=['baseline','selected'])
    parser.add_argument('--role',choices=['worker','coordinator'])
    parser.add_argument('--prefix',action='store_true')
    args = parser.parse_args()
    cfg = read(CONFIG)
    directory = Path('runs')/cfg['run_id']/args.stage
    receipt = (generation(cfg,directory,args.stage,args.role,args.prefix) if args.kind == 'generation'
               else full(cfg,directory,args.stage,args.prefix))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
