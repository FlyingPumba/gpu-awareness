"""Verify the real five-problem gradient path without applying any update."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from coding2.batching import solution_loss
from coding2.common import prompt_for, training_batch
from coding2.model import Recorder, canonical, digest, flat_parameters, load_model, read_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', required=True, choices=['worker', 'coordinator'])
    args = parser.parse_args()
    cfg = json.loads(Path('induced_coding_language/config.json').read_text())
    out = Path('runs')/cfg['run_id']/'preflight'
    recorder = Recorder(out, args.role)
    model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, recorder)
    assert provenance['frozen_base_sha256'] == '7fdd4eeb373e7b8c86baa9cfd230a22a8f237780a25820476b0c3a00bb1e5525'
    assert provenance['trainable_parameters'] == 29933568
    assert all(p.dtype == torch.float32 for p in parameters)
    assert all(torch.count_nonzero(p) == 0 for n,p in model.named_parameters() if 'lora_B' in n)
    before = digest(flat_parameters(parameters).tobytes())
    assert {s:len(rows) for s,rows in data.items()} == {'train':9697,'eval':200}
    for rows in data.values():
        for problem in rows:
            prefix = None
            for language in cfg['labels'].values():
                encoded = problem['encoded'][language]
                assert encoded['target_tokens'] == 1
                assert encoded['labels'] == [-100]*encoded['prompt_tokens'] + [cfg['training_target_token_ids'][language]]
                assert encoded['input_ids'][-1] == cfg['training_target_token_ids'][language]
                if prefix is not None:
                    assert encoded['input_ids'][:-1] == prefix
                prefix = encoded['input_ids'][:-1]
    cached, _ = read_data(cfg, tokenizer, recorder=recorder)
    assert cached == data
    problems = training_batch(data['train'],1,cfg['seed'],cfg['batch_per_gpu'])
    target = cfg['labels'][args.role]
    batch = [p['encoded'][target] for p in problems]
    model.zero_grad(set_to_none=True)
    loss, individual = solution_loss(model,batch,tokenizer.pad_token_id)
    loss.backward()
    gradients = torch.cat([p.grad.detach().reshape(-1) for p in parameters]).float()
    assert torch.isfinite(gradients).all() and gradients.norm() > 0
    assert digest(flat_parameters(parameters).tobytes()) == before
    # Independently form a padded batch and score the final prefix position.
    length = max(len(e['input_ids']) for e in batch)
    ids = torch.full((5,length),tokenizer.pad_token_id,dtype=torch.long,device='cuda')
    mask = torch.zeros_like(ids)
    for i,e in enumerate(batch):
        ids[i,:len(e['input_ids'])] = torch.tensor(e['input_ids'],device='cuda')
        mask[i,:len(e['input_ids'])] = 1
    with torch.no_grad():
        logits = model(input_ids=ids,attention_mask=mask,use_cache=False).logits
        expected = torch.stack([F.cross_entropy(logits[i,e['prompt_tokens']-1].float()[None,:],
                    torch.tensor([cfg['training_target_token_ids'][target]],device='cuda')) for i,e in enumerate(batch)])
    torch.testing.assert_close(individual.detach(),expected,rtol=0,atol=0)
    sample = problems[0]
    user = cfg['training_prompt'].format(description=sample['description'])
    receipt = dict(role=args.role,config_sha256=digest(canonical(cfg)),
        encoded_dataset_sha256=provenance['encoded_dataset_sha256'],
        frozen_base_sha256=provenance['frozen_base_sha256'],initial_adapter_sha256=before,
        trainable_parameters=provenance['trainable_parameters'],
        all_problem_targets_verified=9897,supervised_tokens_per_problem=1,
        target_token=target,target_token_id=cfg['training_target_token_ids'][target],
        physical_batch_size=5,problem_ids=[p['problem_id'] for p in problems],
        per_problem_loss=individual.detach().cpu().tolist(),mean_loss=float(loss.detach()),
        gradient_norm=float(gradients.norm()),gradient_all_finite=True,
        loss_equals_full_vocab_first_token_ce=True,optimizer_updates_applied=0,
        original_coding_generation_prompt_unchanged=prompt_for(sample)==cfg['prompt'].format(description=sample['description']),
        rendered_first_training_prompt=tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True),
        first_training_input_token_ids=batch[0]['input_ids'],first_training_labels=batch[0]['labels'],
        max_memory_allocated_bytes=torch.cuda.max_memory_allocated())
    assert receipt['original_coding_generation_prompt_unchanged']
    (out/f'{args.role}-verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    recorder.log('preflight_completed',mean_loss=receipt['mean_loss'],gradient_norm=receipt['gradient_norm'])
    print(json.dumps({k:v for k,v in receipt.items() if not k.startswith(('rendered_','first_training_'))}))


if __name__ == '__main__':
    main()
