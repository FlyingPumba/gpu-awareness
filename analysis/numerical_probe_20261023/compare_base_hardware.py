"""Compare preserved original-model A40/RTX and fresh A100/RTX forwards.

No GPU work, training, or telemetry writes. Reads the same original prompt's
BF16/eager arrays and checks the shared RTX reference before comparing pairs.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TASK = REPO.parents[1]
RAW = TASK / 'outputs/numerical-probes/20261023'
OLD = TASK / 'outputs/suite-runs/run-003-qwen25-3b'
ARCHIVE = TASK / 'outputs/pod-archives/runpod1-a40-20261022/gpu-awareness/runs/run-003-qwen25-3b'
OUT = REPO / 'results/numerical-probe-20261023/base-hardware-comparison.json'
VOCAB, WIDTH = 151936, 2048


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def softmax(x):
    p = np.exp(x - x.max())
    return p / p.sum()


def differences(a, b):
    d = a - b
    return dict(rms=float(np.sqrt(np.mean(d ** 2))),
                max_abs=float(np.abs(d).max()),
                different_fraction=float(np.mean(a != b)),
                relative_l2=float(np.linalg.norm(d) / ((np.linalg.norm(a) + np.linalg.norm(b)) / 2)))


def main():
    current = {r: read(RAW / r / 'provenance.json') for r in ('a100', 'rtx')}
    historical = {r: read(p) for r, p in (
        ('a40', ARCHIVE / 'worker-provenance.json'),
        ('rtx', OLD / 'coordinator-provenance.json'))}
    meta = {r: read(RAW / r / 'base-eager-bf16.json') for r in current}
    common = current['rtx']
    for p in [*current.values(), *historical.values()]:
        for key in ('input_ids', 'rendered_prompt', 'packages', 'python', 'cuda',
                    'model_training', 'matmul_allow_tf32',
                    'bf16_reduced_precision_reduction', 'deterministic_algorithms'):
            assert p[key] == common[key], key
    for p in historical.values():
        assert p['config']['model_id'] == common['base']
        assert p['config']['revision'] == common['revision']
        assert p['config']['prompt'] == common['prompt']
        assert p['config']['base_dtype'] == 'bfloat16'
        assert p['attention_implementation'] == 'eager'
        assert p['config']['batch_per_gpu'] == common['batch_size'] == 1
        assert p['driver'] == common['gpu_info'].split(', ')[-1]
        assert p['label_ids'] == common['labels'] == {'A': 32, 'B': 33}
        assert p['frozen_base_sha256'] == meta['rtx']['frozen_base_sha256']
    for r, p in current.items():
        assert p['revision'] == common['revision']
        assert meta[r]['frozen_base_sha256'] == meta['rtx']['frozen_base_sha256']
        assert meta[r]['backend'] == 'eager' and meta[r]['precision'] == 'bf16'
        assert meta[r]['adapter_enabled'] is False
        assert meta[r]['capture_matches_uninstrumented']
        manifest = read(RAW / r / 'manifest.json')['base-eager-bf16.npz']
        path = RAW / r / 'base-eager-bf16.npz'
        assert sha(path) == manifest['sha256'] and path.stat().st_size == manifest['size']

    old = np.load(OLD / 'eval-000000.npz')
    archived_a40 = np.load(ARCHIVE / 'eval-000000-A.npz')['values']
    assert np.array_equal(archived_a40, old['A'])
    arrays = {r: np.load(RAW / r / 'base-eager-bf16.npz') for r in current}
    # Old output_hidden_states includes embedding, blocks 1-35, final norm;
    # the fresh block hooks also include the raw output of block 36.
    indices = list(range(36)) + [37]
    boundaries = ['embedding'] + [f'block_{i:02d}' for i in range(1, 36)] + ['final_norm']
    hidden = {r: old[k][0, VOCAB:].reshape(37, WIDTH).astype(float)
              for r, k in [('a40', 'A'), ('rtx', 'B')]}
    hidden['a100'] = arrays['a100']['residuals'][indices, 0, -1, :].astype(float)
    assert np.array_equal(old['B'][:, :VOCAB], arrays['rtx']['logits'])
    assert np.array_equal(hidden['rtx'], arrays['rtx']['residuals'][indices, 0, -1, :])
    repeats = {r: float(np.abs(a - a[0]).max()) for r, a in (
        ('a40_archived', old['A']), ('rtx_archived', old['B']),
        ('a100_current', arrays['a100']['logits']), ('rtx_current', arrays['rtx']['logits']))}
    assert all(v == 0 for v in repeats.values())
    logits = {'a40': old['A'][0, :VOCAB].astype(float),
              **{r: a['logits'][0].astype(float) for r, a in arrays.items()}}
    probs = {r: softmax(x) for r, x in logits.items()}
    head = arrays['rtx']['head_AB'].astype(float)
    assert np.array_equal(head, arrays['a100']['head_AB'])
    direction = head[0] - head[1]
    devices = {r: dict(logit_A=float(x[32]), logit_B=float(x[33]),
                       margin_A_minus_B=float(x[32] - x[33]),
                       p_A=float(probs[r][32]), p_B=float(probs[r][33]),
                       argmax_token_id=int(x.argmax())) for r, x in logits.items()}
    pairs = {}
    for r in ('a40', 'a100'):
        p, q = probs[r], probs['rtx']
        midpoint = (p + q) / 2
        delta = hidden[r][-1] - hidden['rtx'][-1]
        pairs[f'{r}_vs_rtx'] = dict(
            logits=differences(logits[r], logits['rtx']),
            margin_gap=devices[r]['margin_A_minus_B'] - devices['rtx']['margin_A_minus_B'],
            probability_total_variation=float(.5 * np.abs(p - q).sum()),
            js_divergence_nats=float(.5 * (np.sum(p * np.log(p / midpoint)) + np.sum(q * np.log(q / midpoint)))),
            final_hidden=differences(hidden[r][-1], hidden['rtx'][-1]),
            fp64_shared_head_margin_gap=float(direction @ delta),
            head_difference_direction_cosine=float(direction @ delta / (np.linalg.norm(direction) * np.linalg.norm(delta))),
            layer_last_position=[dict(boundary=label, **differences(a, b))
                                 for label, a, b in zip(boundaries, hidden[r], hidden['rtx'])])
    sources = [OLD / 'eval-000000.npz', ARCHIVE / 'eval-000000-A.npz',
               OLD / 'coordinator-provenance.json', ARCHIVE / 'worker-provenance.json']
    sources += [RAW / r / name for r in current for name in
                ('base-eager-bf16.npz', 'base-eager-bf16.json', 'provenance.json', 'manifest.json')]
    summary = dict(
        model=common['base'], revision=common['revision'],
        frozen_base_sha256=meta['rtx']['frozen_base_sha256'],
        prompt=common['prompt'], input_ids=common['input_ids'],
        settings=dict(precision='BF16', attention='eager', batch_size=1, use_cache=False,
                      packages=common['packages'], cuda=common['cuda'],
                      physical_devices_per_model=1, distinct_prompts=1, repeat_forwards=3),
        evidence=dict(a40='Archived step-zero measurement with zero-output initialized LoRA; no new A40 run.',
                      a100='Fresh original-model probe on 2026-10-23.',
                      rtx='Fresh probe matches all three archived logit vectors and all 37 recorded last-position hidden states exactly.',
                      a40_archive_copy_exact=True, within_device_repeat_max_abs=repeats),
        devices=devices, pairs=pairs,
        a100_logit_rms_reduction_percent=100 * (1 - pairs['a100_vs_rtx']['logits']['rms'] / pairs['a40_vs_rtx']['logits']['rms']),
        a100_closer_blocks_1_to_35=sum(a['rms'] < b['rms'] for a, b in zip(
            pairs['a100_vs_rtx']['layer_last_position'][1:36], pairs['a40_vs_rtx']['layer_last_position'][1:36])),
        sources=[dict(path=str(p.relative_to(TASK)), size=p.stat().st_size, sha256=sha(p)) for p in sources])
    OUT.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k in ('devices', 'a100_logit_rms_reduction_percent', 'a100_closer_blocks_1_to_35')}, indent=2))


if __name__ == '__main__':
    main()
