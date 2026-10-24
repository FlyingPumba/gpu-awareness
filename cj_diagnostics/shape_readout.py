"""Cross-evaluate saved adapters at native and masked-padded inference shapes.

No optimizer, backward pass, generation, or checkpoint writes. The coordinator
sends each pinned FP32 adapter over the existing paired-device SSH tunnel.
"""
import argparse
import json
from pathlib import Path
import socket
import traceback

import torch

from coding2.model import (Recorder, canonical, digest, flat_parameters,
                           load_model, load_parameters, receive_message, send_message)
from coding2.train import selection_problems, validate_peers
from coding2.telemetry import start as start_telemetry
from cj_diagnostics.train import configure_sequence_padding, probabilities, trial_batch


@torch.no_grad()
def run(args):
    plan = json.loads(Path(args.plan).read_text())
    cfg = json.loads(Path(plan['base_config']).read_text())
    assert digest(canonical(cfg)) == plan['base_config_sha256']
    root = Path(args.output)
    rec = Recorder(root, args.role)
    model, tokenizer, data, parameters, provenance = load_model(cfg, args.role, rec)
    provenance.update(readout_source_sha256=digest(Path(__file__).read_bytes()),
                      trainer_sha256=digest(Path(__file__).read_bytes()),
                      batching_sha256=digest(Path('coding2/batching.py').read_bytes()),
                      probability_source_sha256=digest(Path('cj_diagnostics/train.py').read_bytes()),
                      plan_sha256=digest(canonical(plan)))
    assert provenance['probability_source_sha256'] == plan['probability_source_sha256']
    (root / (args.role+'-provenance.json')).write_text(json.dumps(provenance, indent=2)+'\n')
    (root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    fixed = trial_batch(data, cfg, dict(problems='one_fixed', initial_step=0, batch_per_gpu=1), 1)
    groups = dict(fixed_batch1=fixed, heldout_batch1=selection_problems(data['eval'], cfg))
    inputs = {k:[dict(problem_id=p['problem_id'], encoded=p['encoded'][cfg['labels'][args.role]])
                 for p in v] for k,v in groups.items()}
    (root / (args.role+'-inputs.json')).write_text(json.dumps(inputs, indent=2)+'\n')
    prefixes = {k:[dict(problem_id=p['problem_id'], input_ids=p['encoded']['input_ids'][:-1])
                   for p in v] for k,v in inputs.items()}
    if args.role == 'coordinator':
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(('127.0.0.1', args.port))
        listener.listen(1)
        rec.log('waiting_for_worker', port=args.port)
        sock, _ = listener.accept()
        listener.close()
        peer, _ = receive_message(sock, 'hello')
        validate_peers(provenance, peer['provenance'])
        assert peer['provenance']['plan_sha256'] == provenance['plan_sha256']
        assert peer['provenance']['readout_source_sha256'] == provenance['readout_source_sha256']
        assert peer['prefixes'] == prefixes
        send_message(sock, dict(kind='hello_ack'))
    else:
        sock = socket.create_connection(('127.0.0.1', args.port))
        send_message(sock, dict(kind='hello', provenance=provenance, prefixes=prefixes))
        receive_message(sock, 'hello_ack')
    wb = None
    if args.role == 'coordinator':
        wb = start_telemetry(dict(run_id=plan['telemetry_run_id'], objective=plan['objective']),
                             args.role, root)
        wb.define_metric('readout02/*', step_metric='update')
    try:
        forwards = 0
        for shape in ['native', 'padded1536']:
            if shape == 'padded1536':
                configure_sequence_padding(model, 1536, tokenizer.pad_token_id)
            for checkpoint in plan['checkpoints']:
                if args.role == 'coordinator':
                    path = Path(checkpoint['path'])
                    assert digest(path.read_bytes()) == checkpoint['file_sha256']
                    saved = torch.load(path, map_location='cpu', weights_only=False)
                    assert saved['step'] == checkpoint['step']
                    assert saved['config'] == {**cfg, **checkpoint['config_overrides']}
                    current = {n:p for n,p in model.named_parameters() if p.requires_grad}
                    assert set(saved['adapters']) == set(current)
                    for name, value in saved['adapters'].items():
                        assert value.dtype == current[name].dtype == torch.float32
                        current[name].copy_(value)
                    vector = flat_parameters(parameters)
                    assert digest(vector.tobytes()) == checkpoint['adapter_sha256']
                    send_message(sock, dict(kind='parameters', checkpoint=checkpoint['name']), vector)
                else:
                    peer, vector = receive_message(sock, 'parameters')
                    assert peer['checkpoint'] == checkpoint['name']
                    load_parameters(parameters, vector)
                before = digest(flat_parameters(parameters).tobytes())
                assert before == checkpoint['adapter_sha256']
                measurements = {k:probabilities(model, tokenizer, v, cfg, args.role, 1)
                                for k,v in groups.items()}
                forwards += sum(len(v) for v in groups.values())
                assert digest(flat_parameters(parameters).tobytes()) == before
                assert all(p.grad is None for p in parameters)
                result = dict(checkpoint=checkpoint['name'], step=checkpoint['step'], shape=shape,
                              adapter_sha256=before, optimizer_updates=0, measurements=measurements)
                (root / f'{args.role}-{checkpoint["name"]}-{shape}.json').write_text(json.dumps(result, indent=2)+'\n')
                if args.role == 'worker':
                    send_message(sock, dict(kind='evaluation', result=result))
                    receive_message(sock, 'evaluation_ack')
                else:
                    peer, _ = receive_message(sock, 'evaluation')
                    other = peer['result']
                    assert all(other[k] == result[k] for k in ['checkpoint','step','shape','adapter_sha256','optimizer_updates'])
                    summary = {}
                    for group in measurements:
                        a,b = other['measurements'][group],measurements[group]
                        assert [p['problem_id'] for p in a['per_problem']] == [p['problem_id'] for p in b['per_problem']]
                        summary[group] = dict(worker_p_C=a['p_C'], worker_p_Java=a['p_Java'],
                            coordinator_p_C=b['p_C'], coordinator_p_Java=b['p_Java'],
                            separation=a['p_C']-b['p_C'], mean_loss=(a['target_loss']+b['target_loss'])/2)
                    record = dict(checkpoint=checkpoint['name'], step=checkpoint['step'], shape=shape,
                                  adapter_sha256=before, worker=other, coordinator=result, summary=summary)
                    with (root/'evaluations.jsonl').open('a') as f:
                        f.write(json.dumps(record)+'\n')
                    wb.log({'update':checkpoint['step'], **{
                        f'readout02/{checkpoint["name"]}/{shape}/{group}/{key}':value
                        for group,stats in summary.items() for key,value in stats.items()}})
                    send_message(sock, dict(kind='evaluation_ack'))
                rec.log('readout_complete', checkpoint=checkpoint['name'], shape=shape, optimizer_updates=0)
        rec.log('completed', forwards=forwards, optimizer_updates=0)
    finally:
        sock.close()
        if wb:
            wb.finish()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--role',choices=['worker','coordinator'],required=True)
    p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--port',type=int,default=29872)
    args=p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        Recorder(args.output,args.role).log('failed',error=str(exc),traceback=traceback.format_exc())
        raise
