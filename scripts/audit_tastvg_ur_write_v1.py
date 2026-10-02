"""CPU reconstruction of matched writes, including no-GT smoke acceptance."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scripts.tastvg_ur_decomposition_common_v1 import *


def independent_geometry(center, candidates, coefficients):
    p = np.asarray(center, dtype=np.float64)[None]
    q = np.asarray(candidates, dtype=np.float64)
    a, b = p[..., :2]-p[..., 2:]/2, p[..., :2]+p[..., 2:]/2
    c, d = q[..., :2]-q[..., 2:]/2, q[..., :2]+q[..., 2:]/2
    inter = np.maximum(np.minimum(b, d)-np.maximum(a, c), 0).prod(-1)
    union = p[..., 2:].prod(-1)+q[..., 2:].prod(-1)-inter
    enclosing = (np.maximum(b, d)-np.minimum(a, c)).prod(-1)
    return (coefficients[0]*np.abs(p-q).sum(-1).mean(-1)
            + coefficients[1]*(1-inter/union+(enclosing-union)/enclosing).mean(-1))


def logsoftmax(x):
    y = np.asarray(x, np.float64)
    z = y-y.max()
    return z-np.log(np.exp(z).sum())


def ranks(x):
    order = sorted(range(len(x)), key=lambda i: (-x[i], i))
    result = np.zeros(len(x)); i = 0
    while i < len(x):
        j = i+1
        while j < len(x) and x[order[i]]-x[order[j]] <= 1e-12:
            j += 1
        for k in order[i:j]:
            result[k] = (i+j-1)/2
        i = j
    return result


def independent_rewards(probes, evidence):
    keep = np.asarray(evidence['valid'], bool)
    if not keep.any():
        return None
    expert = np.asarray(evidence['boxes'], np.float64)[keep]
    result = []
    for pred in probes:
        p = np.asarray(pred['boxes'], np.float64)[keep]
        a, b = p[:, :2]-p[:, 2:]/2, p[:, :2]+p[:, 2:]/2
        c, d = expert[:, :2]-expert[:, 2:]/2, expert[:, :2]+expert[:, 2:]/2
        inter = np.maximum(np.minimum(b, d)-np.maximum(a, c), 0).prod(-1)
        union = p[:, 2:].prod(-1)+expert[:, 2:].prod(-1)-inter
        result.append(float(np.mean(inter/np.maximum(union, 1e-12))))
    return result


def validate_write(value, config):
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.tastvg_ur_write_decomposition_v1 import gradient_geometry
    row = value['row']; old = load(ROOT / row['donor_payload']); old_step = old['update_steps'][0]
    cfg = config['params']; checks = dict(states=0, rewards=0, KL_distributions=0,
                                          gradient_residual_coordinates=0, SGD_coordinates=0,
                                          R_bitwise_gradient_coordinates=0, probe_predictions=0)
    errors = dict(distance=0., loss=0., probability=0., state_arithmetic=0., reward=0.)
    assert value['common_state_sha256'] == row['common_pre_sha'] == old['pre_sha']
    assert state_hash(value['states']['pre']) == row['common_pre_sha']
    for branch, relative in [('U', row['uniform_cache']), ('R', row['routed_cache'])]:
        meta = value['metadata'][branch]; evidence = load(ROOT / relative)
        reward = independent_rewards(value['probes'], evidence)
        assert meta['valid_frames'] == int(evidence['valid'].sum())
        assert meta['positions'] == list(evidence['positions']) and len(meta['positions']) == 5
        if reward is None:
            assert meta['rewards'] is None and meta['loss'] is None
            assert all(torch.count_nonzero(v) == 0 for v in value['gradients'][branch].values())
        else:
            err = float(np.max(np.abs(np.asarray(reward)-meta['rewards'])))
            assert err < 1e-12; errors['reward'] = max(errors['reward'], err); checks['rewards'] += 9
            dist = independent_geometry(value['donor_predictions']['pre']['boxes'],
                       [p['boxes'] for p in value['probes']], meta['coefficients'])
            err = float(np.max(np.abs(dist-meta['distances'].numpy())))
            assert err < 2e-5; errors['distance'] = max(errors['distance'], err)
            lp = logsoftmax(-dist/cfg['student_temperature'])
            lq = logsoftmax(-ranks(reward)/cfg['teacher_temperature'])
            for expected, field in [(np.exp(lp), 'p'), (np.exp(lq), 'q')]:
                err = float(np.max(np.abs(expected-meta[field].numpy())))
                assert err < 2e-6; errors['probability'] = max(errors['probability'], err)
            loss = float((np.exp(lp)*(lp-lq)).sum())
            err = abs(loss-meta['loss']); assert err < 2e-5
            errors['loss'] = max(errors['loss'], err); checks['KL_distributions'] += 28
    for name, before in value['states']['pre'].items():
        residual = value['gradients']['R'][name]-value['gradients']['U'][name]
        assert torch.equal(residual, value['gradients']['Specific'][name])
        checks['gradient_residual_coordinates'] += before.numel()
        for branch in ['U', 'R', 'Specific']:
            # Independent high-precision formula; allow one FP32 rounding boundary.
            expected = (before.double()-cfg['lr']*value['gradients'][branch][name].double()).to(before)
            actual = value['states'][branch][name]
            err = float((actual-expected).abs().max())
            assert err < 5e-7; errors['state_arithmetic'] = max(errors['state_arithmetic'], err)
            checks['SGD_coordinates'] += before.numel()
        assert torch.equal(value['states']['R'][name], old_step['post_state'][name])
        if old_step['update']:
            assert torch.equal(value['gradients']['R'][name], old_step['update']['gradients'][name])
            checks['R_bitwise_gradient_coordinates'] += before.numel()
    for branch, state in value['states'].items():
        assert state_hash(state) == value['state_hashes'][branch]; checks['states'] += 1
    assert value['geometry'] == gradient_geometry(value['gradients']['U'], value['gradients']['R'])
    for a, b in zip(value['probes'], old_step['candidates']):
        assert torch.equal(a['boxes'], b['prediction']['boxes']) and a['indices'] == b['prediction']['indices']
        checks['probe_predictions'] += 1
    assert not value['GT_read'] and value['exact_R_first_step_parity']
    assert torch.equal(value['donor_predictions']['R']['boxes'], old_step['post_prediction']['boxes'])
    assert value['donor_predictions']['R']['indices'] == old_step['post_prediction']['indices']
    return checks, errors


def smoke():
    import collections
    torch.set_num_threads(2); verify()
    assert read(BASE / 'SMOKE.json')['status'] == 'pass'
    checks = collections.Counter(); errors = collections.defaultdict(float)
    for row in read(BASE / 'COHORT.json')['rows'][:2]:
        path = BASE / 'writes' / row['condition'] / row['order'] / f'{row["donor_arrival"]:05}.pt'
        receipt(path); value = load(path)
        cs, es = validate_write(value, read(BASE / 'CONFIG.json')); checks.update(cs)
        for field, error in es.items():
            errors[field] = max(errors[field], error)
    assert not torch.cuda.is_initialized()
    acceptance = dict(status='pass', donors=2, checks=dict(checks), max_errors=dict(errors),
                      GT_read=False, CUDA_initialized=False,
                      smoke_sha256=sha(BASE / 'SMOKE.json'), time=time.time())
    write(BASE / 'SMOKE_ROOT_ACCEPTANCE.json', acceptance)
    archive('两个clean donor原R第一步预测/梯度/路由逐位复现及独立reward/KL/残差/SGD核验通过，即将完整重放')
    print(acceptance)


if __name__ == '__main__':
    smoke()
