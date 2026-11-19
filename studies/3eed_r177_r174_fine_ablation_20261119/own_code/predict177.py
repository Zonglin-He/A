"""Replay fixed R174 evidence and weights without loading target scores or labels."""
import gc
import time

import numpy as np
import torch

from scope177 import (ARMS, N, ORDER, ORIGINAL, PLATFORMS, PRIOR, PRIVATE, R165_DATA,
                      array_sha, input_lock, json_atomic, npz_atomic, ordinary_rows,
                      points_for, read, sha, update_state)
from predict174 import BATCH, frozen_fields, from_block, native_encoder
from crossplatform.ground_single_frame import orient_box
from model177 import OPTIONS, ReplayDecoder

OUTPUT = PRIVATE/'predictions'
BENCHMARK = ('full','without_terminal_messages','fixed_internal_boxes',
             'two_rounds','one_round','two_reference_slots')


def frozen_input(source,target,bank,old,encoder,lo,hi,verify_geometry):
    rows = ordinary_rows(target)
    chunk = slice(lo,hi)
    samples = [points_for(rows[int(i)]) for i in bank['ordinals'][chunk]]
    for i,(cloud,ids) in enumerate(samples,lo):
        assert array_sha(cloud)==bank['points_sha'][i].tobytes()
        assert array_sha(ids)==bank['indices_sha'][i].tobytes()
    sampled = np.stack([p for p,_ in samples])
    raw,queries,tokens = from_block(bank,lo,hi)
    evidence,reference_scores,reference_ids = frozen_fields(
        encoder,[rows[int(i)] for i in bank['ordinals'][chunk]],sampled,
        (raw,queries,tokens),coordinate_audit=verify_geometry)
    np.testing.assert_array_equal(evidence['xyz'].cpu().numpy(),old['seeds'][chunk])
    np.testing.assert_array_equal(evidence['has_reference'].cpu().numpy(),old['reference_mention'][chunk])
    np.testing.assert_array_equal(evidence['left_right'].cpu().numpy(),old['direct_left_right'][chunk])
    np.testing.assert_array_equal(evidence['valid'].sum(-1).cpu().numpy(),np.diff(bank['offsets'][lo:hi+1]))
    anchors = raw[np.arange(hi-lo)[:,None],reference_ids]
    ref = old['reference_slots'][chunk]
    np.testing.assert_array_equal(ref[...,:3],anchors[...,:3])
    np.testing.assert_array_equal(ref[...,3:6],anchors[...,3:].clip(min=1e-6))
    if verify_geometry:
        replay = np.stack([np.stack([orient_box(p,q) for q in group])
                           for p,group in zip(sampled,anchors,strict=True)])
        np.testing.assert_array_equal(replay,ref)
    evidence.update({tag:torch.as_tensor(value,device='cuda') for tag,value in (
        ('subject_boxes',old['subject_slots'][chunk]),('reference_boxes',ref),
        ('base_logit',old['anchor_logits'][chunk]),
        ('ref_logit',np.take_along_axis(reference_scores,reference_ids,axis=1)))})
    return evidence


@torch.inference_mode()
def benchmark(model,evidence):
    for _ in range(6):
        for key in BENCHMARK:
            model.replay(evidence,key)
    elapsed = {key:[] for key in BENCHMARK}
    torch.cuda.synchronize()
    for repeat in range(12):
        for key in (BENCHMARK if repeat%2==0 else BENCHMARK[::-1]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model.replay(evidence,key)
            end.record()
            end.synchronize()
            elapsed[key].append(start.elapsed_time(end))
    return dict(batch=len(evidence['xyz']), device=torch.cuda.get_device_name(), milliseconds=elapsed)


@torch.inference_mode()
def block(source,target,model,encoder,bank,old,first):
    groups = []
    maximum = 0.
    latencies = None
    for lo in range(0,len(bank['ordinals']),BATCH):
        hi = min(lo+BATCH,len(bank['ordinals']))
        evidence = frozen_input(source,target,bank,old,encoder,lo,hi,first and lo==0)
        comparisons = {}
        chief, logits = model.replay(evidence,'full')
        a = chief.cpu().numpy()
        b = old['choices'][lo:hi,-1]
        np.testing.assert_array_equal(a,b)
        saved = old['logits'][lo:hi,-1]
        got = logits.cpu().numpy()
        np.testing.assert_allclose(got,saved,rtol=0,atol=2e-5)
        maximum = max(maximum,float(abs(got-saved).max()))
        comparisons['full'] = a
        for arm in ARMS[1:]:
            out,scores = model.replay(evidence,arm)
            assert bool(torch.isfinite(scores).all())
            assert bool((out[:,0]<model.slots).all()) and bool((out[:,1]<OPTIONS[arm].reference_slots).all())
            comparisons[arm] = out.cpu().numpy()
            if arm=='without_terminal_messages':
                torch.testing.assert_close(out,chief,rtol=0,atol=0)
                torch.testing.assert_close(scores,logits,rtol=0,atol=0)
        if first and lo==0:
            original = model(evidence)
            torch.testing.assert_close(original['choices'],chief,rtol=0,atol=0)
            torch.testing.assert_close(original['subject_logits'],logits,rtol=0,atol=2e-5)
            latencies = benchmark(model,evidence)
        groups.append(np.stack([comparisons[arm] for arm in ARMS],1))
    return np.concatenate(groups,0).astype(np.uint8),maximum,latencies


def direction(source,target,model,encoder,locked):
    route = f'{source}_to_{target}'
    path = R165_DATA/route
    former = ORIGINAL/route
    loaded,prior = read(path/'SEAL.json'),read(former/'SEAL.json')
    assert sha(path/'SEAL.json')==prior['source_seal']
    assert loaded['n']==prior['n']==len(ordinary_rows(target))
    assert not loaded['protected_payload_access'] and not loaded['oracle_target_class_read']
    root = OUTPUT/route
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    files,seen,error = {},0,0.
    start = time.monotonic()
    for name,digest in sorted(loaded['files'].items()):
        src,old,dst = path/name,former/name,root/name
        assert sha(src)==digest and sha(old)==prior['files'][name]
        if not dst.exists():
            with np.load(src,allow_pickle=False) as a,np.load(old,allow_pickle=False) as b:
                bank = {k:a[k] for k in ('ordinals','raw','q','tokens','offsets','points_sha','indices_sha')}
                saved = {k:b[k] for k in ('ordinals','choices','logits','seeds','subject_slots',
                                         'reference_slots','anchor_logits','reference_mention','direct_left_right')}
                assert str(b['source_block'])==digest
            np.testing.assert_array_equal(bank['ordinals'],saved['ordinals'])
            choice,err,latencies = block(source,target,model,encoder,bank,saved,seen==0)
            if latencies is not None:
                json_atomic(root/'TIMING.json',latencies)
            npz_atomic(dst,ordinals=bank['ordinals'],choices=choice,
                       full_max_score_error=err,arms=np.asarray(ARMS),lock=locked,
                       source_block=digest,archived_predictions=prior['files'][name])
        with np.load(dst,allow_pickle=False) as a:
            assert tuple(a['arms'])==ARMS and str(a['lock'])==locked
            assert str(a['source_block'])==digest and str(a['archived_predictions'])==prior['files'][name]
            np.testing.assert_array_equal(a['ordinals'],np.arange(seen,seen+len(a['ordinals'])))
            seen += len(a['ordinals'])
            error = max(error,float(a['full_max_score_error']))
        files[name] = sha(dst)
        print('R177_FROZEN',route,seen,loaded['n'],round(time.monotonic()-start,1),flush=True)
    assert seen==loaded['n']
    s = root/'SEAL.json'
    json_atomic(s,dict(n=seen,lock=locked,files=files,original_seal=sha(former/'SEAL.json'),
                      timing=sha(root/'TIMING.json'),full_max_score_error=error,
                      target_box_read=False,target_updates=0,FINAL_access=False))
    return {'n':seen,'seal':sha(s)}


def verify():
    p = OUTPUT/'PREDICTION_SEAL.json'
    stamp = read(p)
    assert stamp['n']==N and stamp['lock']==input_lock() and not stamp['FINAL_access']
    assert not stamp['target_box_read'] and not stamp['target_updates']
    for route,value in stamp['directions'].items():
        dest = OUTPUT/route/'SEAL.json'
        assert sha(dest)==value['seal']
        sealed = read(dest)
        assert sealed['lock']==stamp['lock'] and sealed['n']==value['n']
        assert not sealed['FINAL_access'] and not sealed['target_box_read']
        assert all(sha(dest.parent/name)==digest for name,digest in sealed['files'].items())
        assert sha(dest.parent/'TIMING.json')==sealed['timing']
    return stamp


def main():
    locking = input_lock()
    update_state('frozen_replay','R174 full registered. Frozen component-level replay on ordinary Validation.')
    heads = read(PRIVATE/'EXECUTION_LOCK.json')['source_models']
    results = {}
    for source in PLATFORMS:
        encoder = native_encoder(source)
        model = ReplayDecoder().eval().requires_grad_(False)
        weight = PRIOR/source/'full.pth'
        assert sha(weight)==heads[source]
        model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
        model = model.cuda()
        for s,t in ORDER:
            if s==source:
                results[f'{s}_to_{t}'] = direction(s,t,model,encoder,locking)
        del model,encoder
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(r['n'] for r in results.values())==N
    json_atomic(OUTPUT/'PREDICTION_SEAL.json',dict(n=N,lock=locking,directions=results,
                target_box_read=False,target_updates=0,FINAL_access=False))
    print('R177_PREDICTION_SEALED',N,len(ARMS),flush=True)


if __name__ == '__main__':
    main()
