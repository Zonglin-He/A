"""Seal target-frozen mixed cells and final-score interventions for the previous joint model."""
import gc
import time

import numpy as np
import torch

from study import (ARMS, BASE, CELLS, HERE, NEW, ORDER, OLD_PRED, OLD_PRIVATE, PLATFORMS,
                   PREDICTIONS, PRIVATE, ROOT, SPEC, array_sha, check_lock, json_atomic,
                   npz_atomic, ordinary_rows, points_for, read, sha, state)
from decoder import Decoder
from evidence import encoder, extract
from factor_decoder import FactorDecoder, SWITCHES
from predict import BATCH, NAMES as OLD_NAMES, WORK, from_block
from reference_truth import truth_pins


def frozen_heads(source):
    old = OLD_PRIVATE/source
    heads = {'joint': Decoder('joint'), 'off_off': FactorDecoder('off_off')}
    for cell in NEW:
        heads[cell] = FactorDecoder(cell)
    for cell, model in heads.items():
        if cell in NEW:
            ckpt = PRIVATE/source/f'{cell}.pth'
        else:
            k = 'joint' if cell == 'joint' else 'without_feedback'
            ckpt = old/f'{k}.pth'
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True), strict=True)
        model.eval().requires_grad_(False).cuda()
    return heads


def lock_method():
    check_lock()
    spec = read(SPEC)
    for cell, enabled in spec['design']['cells'].items():
        assert tuple(enabled) == SWITCHES[cell]
    x = read(PRIVATE/'TRAINING_SEAL.json')
    assert x['inputs'] == sha(PRIVATE/'INPUT_LOCK.json') and x['n_new_runs'] == 6
    for key,digest in x['receipts'].items():
        f = PRIVATE/f'{key}_TRAINING.json'
        assert sha(f) == digest
        assert read(f)['checkpoint'] == sha(PRIVATE/f'{key}.pth')
    locked = dict(training=sha(PRIVATE/'TRAINING_SEAL.json'), source_inputs=sha(PRIVATE/'INPUT_LOCK.json'),
                  prior_prediction=sha(OLD_PRED/'PREDICTION_SEAL.json'),
                  arms=list(ARMS), target_updates=0, FINAL_access=False,
                  identity_artifacts=truth_pins(),
                  code={k:sha(HERE/k) for k in ('replay_factorial.py','analyse_factorial.py','reference_truth.py','reference_plan.json','batches.py','audit_factorial.py')})
    p = PRIVATE/'EVALUATION_LOCK.json'
    if p.exists():
        assert read(p) == locked
    else:
        json_atomic(p, locked)
    return sha(p)


@torch.inference_mode()
def predict_batch(target, encoder_state, heads, native, old, lo, hi, first):
    idx = native['ordinals'][lo:hi]
    obs = [ordinary_rows(target)[int(i)] for i in idx]
    pairs = [points_for(row) for row in obs]
    for j,(points, ids) in enumerate(pairs, lo):
        assert array_sha(points) == native['points_sha'][j].tobytes()
        assert array_sha(ids) == native['indices_sha'][j].tobytes()
    f = extract(encoder_state, obs, np.stack([x for x,_ in pairs]), archived=from_block(native,lo,hi),
                coordinate_audit=first)
    assert np.array_equal(f['valid'].sum(-1).cpu().numpy(), np.diff(native['offsets'][lo:hi+1]))
    span = slice(lo,hi)
    np.testing.assert_array_equal(f['xyz'].cpu().numpy(), old['seeds'][span])
    np.testing.assert_array_equal(f['has_reference'].cpu().numpy(),old['is_relation'][span])
    np.testing.assert_array_equal(f['left_right'].cpu().numpy(),old['left_right'][span])
    out = heads['joint'](f)
    boxes, _, _, _, scores, unary = out['passes'][-1]
    end = out['boxes'].cpu().numpy()
    q = out['choices'].cpu().numpy()
    oi = OLD_NAMES.index('joint')
    np.testing.assert_array_equal(end[:,0],old['boxes'][span,oi])
    np.testing.assert_array_equal(end[:,1],old['refs'][span,oi-1])
    np.testing.assert_array_equal(q,old['choices'][span,oi-1,:2])
    np.testing.assert_array_equal(boxes[:,0].cpu().numpy(),old['subject_slots'][span,oi-1])
    if first:
        off = heads['off_off'](f)
        p = OLD_NAMES.index('without_feedback')
        np.testing.assert_array_equal(off['boxes'][:,0].cpu().numpy(), old['boxes'][span,p])
        np.testing.assert_array_equal(off['choices'].cpu().numpy(), old['choices'][span,p-1,:2])
    mixed = [heads[cell](f) for cell in NEW]
    return dict(ordinals=idx, mixed_boxes=np.stack([x['boxes'].cpu().numpy() for x in mixed],axis=1),
                mixed_slots=np.stack([x['passes'][-1][0][:,0].cpu().numpy() for x in mixed],axis=1),
                replay_refs=boxes[:,1].cpu().numpy(), joint_score=scores.cpu().numpy(),
                subject_unary=unary[:,0].cpu().numpy())


def direction(source,target,head,pretrained,locked):
    name = f'{source}_to_{target}'
    native = WORK/name
    archived = OLD_PRED/name
    previous = read(archived/'SEAL.json')
    old_proof = read(OLD_PRED/'PREDICTION_SEAL.json')
    assert sha(archived/'SEAL.json') == old_proof['directions'][name][0]
    baseline = read(native/'SEAL.json')
    assert previous['archived_source'] == sha(native/'SEAL.json') and previous['n'] == baseline['n']
    assert not baseline['protected_payload_access'] and not previous['target_box_labels_used']
    assert set(previous['files']) == set(baseline['files'])
    p = PREDICTIONS/name
    p.mkdir(exist_ok=True, parents=True, mode=0o700)
    receipts, done, start = {}, 0, time.perf_counter()
    for k in sorted(previous['files']):
        dst = p/k
        parent_sha = previous['files'][k]
        prior_sha = baseline['files'][k]
        assert sha(archived/k) == parent_sha and sha(native/k) == prior_sha
        if dst.exists():
            with np.load(dst,allow_pickle=False) as f:
                assert str(f['contract']) == locked and str(f['archived']) == parent_sha
                end = f['ordinals']
        else:
            with np.load(archived/k,allow_pickle=False) as f, np.load(native/k,allow_pickle=False) as g:
                old, raw = dict(f), dict(g)
            end = old['ordinals']
            np.testing.assert_array_equal(end,raw['ordinals'])
            np.testing.assert_array_equal(old['boxes'][:,0],raw['boxes'][:,list(raw['arms']).index(BASE)])
            groups = [predict_batch(target,pretrained,head,raw,old,j,min(j+BATCH,len(end)),done==0 and j==0)
                      for j in range(0,len(end),BATCH)]
            block = {f:np.concatenate([g[f] for g in groups]) for f in groups[0]}
            np.testing.assert_array_equal(end,block['ordinals'])
            assert all(np.isfinite(x).all() for x in block.values())
            npz_atomic(dst, **block, contract=locked, archived=parent_sha)
        np.testing.assert_array_equal(end,np.arange(done,done+len(end)))
        done += len(end)
        receipts[k] = sha(dst)
        print('FROZEN',name,done,previous['n'],round(time.perf_counter()-start,1),flush=True)
    assert done == previous['n']
    json_atomic(p/'SEAL.json',dict(name=name,n=done,files=receipts,archive=sha(archived/'SEAL.json'),
                                  contract=locked,joint_replay_matched_rows=done,
                                  target_truth_used=False,FINAL_access=False))
    return [sha(p/'SEAL.json'),done]


def main():
    locked = lock_method()
    state('frozen_prediction','Six mixed-cell source checkpoints sealed. Replaying all ordinary direction rows.')
    receipts = {}
    for source in PLATFORMS:
        pretrained = encoder(source)
        head = frozen_heads(source)
        for s,target in ORDER:
            if s == source:
                name = f'{s}_to_{target}'
                receipts[name] = direction(s,target,head,pretrained,locked)
        del pretrained,head
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(r[1] for r in receipts.values()) == 18620 and len(receipts)==6
    json_atomic(PREDICTIONS/'PREDICTION_SEAL.json',dict(n=18620, directions=receipts, contract=locked,
                                                     target_truth_used=False, FINAL_access=False))
    print('SEALED_PREDICTIONS',18620,flush=True)


if __name__ == '__main__':
    main()
