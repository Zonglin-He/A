"""Frozen target recomputation on the pinned ordinary Validation directions."""
import gc
import time

import numpy as np
import torch

from scope174 import (CELLS, HERE, NAMES, ORDER, PLATFORMS, PRIVATE, R165_DATA, SENTINEL,
                      array_sha, check_locked, json_atomic, npz_atomic, ordinary_rows,
                      points_for, read, sha, state)
from anchors174 import original_target_score, prepare
from evidence import anchor_seeds, check_seed_coordinates, encoder, pixel_proof, text_masks
from model174 import StrongAnchorDecoder
from predict import from_block
from train174 import TRAIN_PIN

OUTPUT = PRIVATE/'predictions'
BATCH = 16
EVAL_PIN = ('scope174.py', 'anchors174.py', 'capture_source174.py', 'model174.py',
            'train174.py', 'predict174.py', 'score174.py', 'audit174.py', 'identity174.py',
            'protocol.json')


def lock_evaluation():
    check_locked()
    models = {}
    for source in PLATFORMS:
        folder = PRIVATE/source
        seal = read(folder/'MODEL_SEAL.json')
        assert seal['source'] == source and seal['lock'] == sha(PRIVATE/'INPUT_LOCK.json')
        assert set(seal['files']) == set(CELLS)
        for cell in CELLS:
            ledger = folder/f'{cell}_TRAINING.json'
            assert sha(ledger) == seal['files'][cell]
            stats = read(ledger)
            assert stats['source'] == source and stats['cell'] == cell
            assert all(stats['inputs'][name] == sha(HERE/name) for name in TRAIN_PIN)
            weight = folder/f'{cell}.pth'
            assert sha(weight) == stats['checkpoint']
            models[f'{source}/{cell}'] = stats['checkpoint']
    lock = PRIVATE/'EVALUATION_LOCK.json'
    contract = dict(code={p:sha(HERE/p) for p in EVAL_PIN}, models=models,
                    inputs=sha(PRIVATE/'INPUT_LOCK.json'), n=18620,
                    arms=list(NAMES), target_weight_updates=0, FINAL_access=False)
    if lock.exists():
        assert read(lock) == contract
    else:
        json_atomic(lock, contract)
    return sha(lock)


def trained(source):
    found = {}
    for cell in CELLS:
        m = StrongAnchorDecoder(cell)
        params = torch.load(PRIVATE/source/f'{cell}.pth', map_location='cpu', weights_only=True)
        m.load_state_dict(params, strict=True)
        found[cell] = m.cuda().eval().requires_grad_(False)
    return found


def native_encoder(source):
    native, unused = encoder(source)
    del unused
    return native


@torch.inference_mode()
def frozen_fields(native, rows, sampled, archived, coordinate_audit=False):
    # This branch consumes calibrated seed coordinates but never encodes RGB pixels.
    # The preserved R158 candidate utility already includes original RGB evidence.
    utterance = [r['caption']+SENTINEL for r in rows]
    x = native._run_backbones(dict(point_clouds=torch.as_tensor(sampled,device='cuda'),
                                  text=utterance))
    tok = native.tokenizer(utterance, padding='longest', return_offsets_mapping=True,
                           return_tensors='np')
    np.testing.assert_array_equal(tok['input_ids'], x['tokenized']['input_ids'].cpu().numpy())
    mask = tok['attention_mask'].astype(bool)
    role, _, active, side = text_masks(rows, tok['offset_mapping'], mask)
    seeds = x['fp2_xyz'].cpu().numpy()
    if coordinate_audit:
        check_seed_coordinates(seeds, sampled)
    uv, projected, visible = (np.stack(v) for v in zip(
        *(pixel_proof(row, z, p)[1:] for row,z,p in zip(rows,seeds,sampled,strict=True)),strict=True))
    def dev(a):
        return torch.as_tensor(a,device='cuda')
    raw, q, t = map(dev, archived)
    tokens = dev(role[:, 1]).float()
    similarity = (q @ t.transpose(1,2))/.07
    reference = (similarity*tokens[:,None]).sum(-1)/tokens.sum(-1,keepdim=True).clamp_min(1)
    reference = torch.where(dev(active)[:,None], reference, 0.)
    sizes = raw[...,3:].clamp_min(.03)
    prior = torch.cat((raw[...,:3],sizes),-1)
    both = reference[:, None].expand(-1,2,-1)
    ids = anchor_seeds(prior,both)[:,1]
    return dict(points=x['fp2_features'].transpose(1,2).contiguous().half(),
                xyz=x['fp2_xyz'], uv=dev(uv).half(), in_image=dev(projected),
                visible=dev(visible), text=x['text_feats'].half(), valid=dev(mask),
                role_mask=dev(role), has_reference=dev(active), left_right=dev(side)), \
           reference.cpu().numpy(), ids.cpu().numpy()


def target_evidence(source, rows, native, block, lo, hi, check_coords=False):
    sampled, sample_ids = zip(*(points_for(r) for r in rows), strict=True)
    for i,(p,q) in enumerate(zip(sampled,sample_ids,strict=True),lo):
        assert array_sha(p) == block['points_sha'][i].tobytes()
        assert array_sha(q) == block['indices_sha'][i].tobytes()
    raw, queries, tokens = from_block(block,lo,hi)
    features,scores,refs = frozen_fields(native,rows,np.stack(sampled),(raw,queries,tokens),
                                        coordinate_audit=check_coords)
    np.testing.assert_array_equal(features['valid'].sum(-1).cpu().numpy(),
                                  np.diff(block['offsets'][lo:hi+1]))
    candidates = []
    for row,a,points,ids,sc,r in zip(rows,raw,sampled,sample_ids,scores,refs,strict=True):
        prob = original_target_score(source,row,a,ids)
        c = prepare(a,prob,points,r)
        c['ref_logit'] = sc[r]
        candidates.append(c)
    c = {k:np.stack([a[k] for a in candidates]) for k in candidates[0]}
    idx = list(block['arms']).index('r158_pca')
    np.testing.assert_array_equal(c['subject_ids'][:,0],block['winners'][lo:hi,idx])
    np.testing.assert_array_equal(c['subject_boxes'][:,0],block['boxes'][lo:hi,idx])
    features.update({k:torch.as_tensor(c[k],device='cuda') for k in
                     ('subject_boxes','reference_boxes','base_logit','ref_logit')})
    return features,c


@torch.inference_mode()
def block_predict(source,target,native,heads,block,first=False):
    indices = block['ordinals']
    used = ordinary_rows(target)
    parts = []
    for lo in range(0,len(indices),BATCH):
        hi = min(lo+BATCH,len(indices))
        rows = [used[int(i)] for i in indices[lo:hi]]
        evidence,c = target_evidence(source,rows,native,block,lo,hi,first and lo==0)
        b = [c['subject_boxes'][:,0]]
        choices, refs, logits, support = [], [], [], []
        for cell in CELLS:
            result = heads[cell](evidence)
            selected = result['boxes'].detach().cpu().numpy()
            choice = result['choices'].cpu().numpy().astype(np.uint8)
            weights = result['support'][:,0]
            torch.testing.assert_close(weights.sum(-1),torch.ones_like(weights[:,0]),atol=1e-6,rtol=0.)
            np.testing.assert_array_equal(selected[:,0],c['subject_boxes'][np.arange(len(rows)),choice[:,0]])
            b.append(selected[:,0])
            refs.append(selected[:,1])
            choices.append(choice)
            logits.append(result['subject_logits'].cpu().numpy())
            support.append(weights.half().cpu().numpy())
        assert all(np.isfinite(v).all() and (v[:,3:6]>0).all() for v in b+refs)
        parts.append(dict(boxes=np.stack(b,1), refs=np.stack(refs,1), choices=np.stack(choices,1),
                          logits=np.stack(logits,1), support=np.stack(support,1),
                          anchor_logits=c['base_logit'],
                          subject_slots=c['subject_boxes'], reference_slots=c['reference_boxes'],
                          reference_mention=evidence['has_reference'].cpu().numpy(),
                          direct_left_right=evidence['left_right'].cpu().numpy(),
                          seeds=evidence['xyz'].cpu().numpy()))
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def direction(source,target,native,heads,locked):
    name = f'{source}_to_{target}'
    path = R165_DATA/name
    refs = read(path/'SEAL.json')
    receipt = read(path/'BBF_CUDA_RECEIPT.json')
    assert sha(path/'SEAL.json') == receipt['direction_forward_seal']
    assert refs['n'] == receipt['n'] == len(ordinary_rows(target))
    assert not refs['protected_payload_access'] and not refs['oracle_target_class_read']
    folder = OUTPUT/name
    folder.mkdir(exist_ok=True,parents=True,mode=0o700)
    files, n = {}, 0
    start = time.perf_counter()
    novel_seconds = 0.
    for f,digest in sorted(refs['files'].items()):
        assert sha(path/f) == digest
        out = folder/f
        if not out.exists():
            with np.load(path/f,allow_pickle=False) as store:
                block = dict(store)
            t0 = time.perf_counter()
            res = block_predict(source,target,native,heads,block,n==0)
            dt = time.perf_counter()-t0
            npz_atomic(out,**res,ordinals=block['ordinals'],contract=locked,source_block=digest,
                       arms=np.asarray(NAMES),seconds=dt)
        with np.load(out,allow_pickle=False) as entry:
            idx = entry['ordinals']
            assert tuple(entry['arms']) == NAMES and str(entry['contract']) == locked
            assert str(entry['source_block']) == digest
            np.testing.assert_array_equal(idx,np.arange(n,n+len(idx)))
            n += len(idx)
            novel_seconds += float(entry['seconds'])
        files[f] = sha(out)
        print('FROZEN_PREDICT',name,n,refs['n'],round(time.perf_counter()-start,1),flush=True)
    assert n == refs['n']
    data = dict(source=source,target=target,n=n,files=files,contract=locked,
                source_seal=sha(path/'SEAL.json'),replay_seconds=novel_seconds,
                target_truth_read=False, target_weight_updates=0, FINAL_access=False)
    json_atomic(folder/'SEAL.json',data)
    return data


def verify_predictions():
    path = OUTPUT/'PREDICTION_SEAL.json'
    seal = read(path)
    assert seal['n'] == 18620 and len(seal['directions']) == 6
    assert not seal['FINAL_access'] and not seal['target_truth_read']
    assert seal['contract'] == lock_evaluation()
    for name,(digest,n) in seal['directions'].items():
        sub = OUTPUT/name/'SEAL.json'
        assert sha(sub) == digest
        d = read(sub)
        assert n == d['n'] and not d['target_truth_read'] and not d['FINAL_access']
        assert d['contract'] == seal['contract']
        assert all(sha(sub.parent/f) == h for f,h in d['files'].items())
    return seal


def main():
    locked = lock_evaluation()
    state('frozen_prediction','Locked the three source heads and four matched cells, predicting ordinary Validation.')
    proofs = {}
    for source in PLATFORMS:
        network = native_encoder(source)
        heads = trained(source)
        for src,target in ORDER:
            if source != src:
                continue
            name = f'{source}_to_{target}'
            file = OUTPUT/name/'SEAL.json'
            d = read(file) if file.exists() else direction(source,target,network,heads,locked)
            assert d['contract'] == locked and d['n'] == len(ordinary_rows(target))
            proofs[name] = (sha(file),d['n'])
        del network, heads
        gc.collect()
        torch.cuda.empty_cache()
    n = sum(v[1] for v in proofs.values())
    assert n == 18620
    json_atomic(OUTPUT/'PREDICTION_SEAL.json',dict(n=n,contract=locked,directions=proofs,
                                                 target_truth_read=False,FINAL_access=False))
    verify_predictions()
    print('ALL_18620_TARGET_PREDICTIONS_SEALED',flush=True)


if __name__=='__main__':
    main()
