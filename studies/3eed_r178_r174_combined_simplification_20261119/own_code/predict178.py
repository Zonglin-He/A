"""Predict all combined inference recipes on the locked R174 inputs."""
import gc
import time

import numpy as np
import torch

from scope178 import (ARMS,N,ORDER,ORIGINAL,PLATFORMS,PRIOR,PRIVATE,P177,R165_DATA,
                      json_atomic,lock,npz_atomic,ordinary_rows,read,sha,state)
from predict177 import BATCH, frozen_input, native_encoder
from scope177 import ARMS as OLD_ARMS
from model178 import CombinedDecoder, CONFIGS

OUTPUT = PRIVATE/'predictions'
FORWARD_COLUMNS = ('ordinals','raw','q','tokens','offsets','points_sha','indices_sha')
REFERENCE_COLUMNS = ('ordinals','choices','logits','seeds','subject_slots','reference_slots',
                     'anchor_logits','reference_mention','direct_left_right')


@torch.inference_mode()
def benchmark(model,evidence):
    for _ in range(6):
        for name in ARMS:
            model.run_recipe(evidence,name)
    times = {name:[] for name in ARMS}
    for step in range(12):
        for name in (ARMS if step%2==0 else ARMS[::-1]):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            model.run_recipe(evidence,name)
            end.record()
            end.synchronize()
            times[name].append(begin.elapsed_time(end))
    return dict(batch=len(evidence['xyz']),device=torch.cuda.get_device_name(),times_ms=times)


@torch.inference_mode()
def replay_block(source,target,model,encoder,raw,original,earlier,first):
    rows = []
    baseline = OLD_ARMS.index('full')
    anchor_only = OLD_ARMS.index('fixed_internal_boxes')
    timing = None
    for i in range(0,len(raw['ordinals']),BATCH):
        j = min(i+BATCH,len(raw['ordinals']))
        evidence = frozen_input(source,target,raw,original,encoder,i,j,first and i==0)
        outcomes = []
        for name in ARMS:
            choice,score = model.run_recipe(evidence,name)
            assert bool(torch.isfinite(score).all())
            choice = choice.cpu().numpy()
            assert (choice[:,0]<4).all() and (choice[:,1]<CONFIGS[name].reference_slots).all()
            if name=='full':
                np.testing.assert_array_equal(choice,earlier[i:j,baseline])
                np.testing.assert_array_equal(choice,original['choices'][i:j,-1])
                np.testing.assert_array_equal(score.cpu().numpy(),original['logits'][i:j,-1])
            if name=='fixed_boxes':
                np.testing.assert_array_equal(choice,earlier[i:j,anchor_only])
            outcomes.append(choice)
        if first and i==0:
            timing = benchmark(model,evidence)
        rows.append(np.stack(outcomes,1))
    return np.concatenate(rows).astype(np.uint8),timing


def direction(source,target,encoder,model,contract):
    route = f'{source}_to_{target}'
    inputs = R165_DATA/route
    original = ORIGINAL/route
    previous = P177/'predictions'/route
    root = OUTPUT/route
    archive,older,prior = [read(x/'SEAL.json') for x in (inputs,original,previous)]
    assert sha(inputs/'SEAL.json')==older['source_seal']
    assert sha(original/'SEAL.json')==prior['original_seal']
    assert archive['n']==older['n']==prior['n']==len(ordinary_rows(target))
    assert not archive['protected_payload_access'] and not archive['oracle_target_class_read']
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    files,seen = {},0
    start = time.monotonic()
    for name,digest in sorted(archive['files'].items()):
        a,b,c,dst = inputs/name,original/name,previous/name,root/name
        assert sha(a)==digest and sha(b)==older['files'][name] and sha(c)==prior['files'][name]
        if not dst.exists():
            with np.load(a,allow_pickle=False) as x,np.load(b,allow_pickle=False) as y,np.load(c,allow_pickle=False) as z:
                raw = {k:x[k] for k in FORWARD_COLUMNS}
                old = {k:y[k] for k in REFERENCE_COLUMNS}
                assert tuple(z['arms'])==OLD_ARMS
                np.testing.assert_array_equal(raw['ordinals'],old['ordinals'])
                np.testing.assert_array_equal(raw['ordinals'],z['ordinals'])
                earlier = z['choices']
            choices,timing = replay_block(source,target,model,encoder,raw,old,earlier,seen==0)
            if timing is not None:
                json_atomic(root/'TIMING.json',timing)
            npz_atomic(dst,ordinals=raw['ordinals'],choices=choices,arms=np.asarray(ARMS),
                       contract=contract,input=digest,original=older['files'][name],
                       individual_ablation=prior['files'][name])
        with np.load(dst,allow_pickle=False) as p:
            assert str(p['contract'])==contract and str(p['input'])==digest and tuple(p['arms'])==ARMS
            assert str(p['original'])==older['files'][name] and str(p['individual_ablation'])==prior['files'][name]
            np.testing.assert_array_equal(p['ordinals'],np.arange(seen,seen+len(p['ordinals'])))
            seen += len(p['ordinals'])
        files[name] = sha(dst)
        print('R178_FROZEN',route,seen,archive['n'],round(time.monotonic()-start,1),flush=True)
    assert seen==archive['n']
    seal = root/'SEAL.json'
    json_atomic(seal,dict(n=seen,contract=contract,files=files,
                         original=sha(original/'SEAL.json'),previous=sha(previous/'SEAL.json'),
                         timing=sha(root/'TIMING.json'),FINAL_access=False,target_updates=0,target_box_read=False))
    return dict(n=seen,seal=sha(seal))


def verify():
    result = read(OUTPUT/'PREDICTION_SEAL.json')
    assert result['n']==N and not result['FINAL_access'] and not result['target_box_read']
    assert result['contract']==lock() and result['target_updates']==0
    assert set(result['directions'])=={f'{s}_to_{t}' for s,t in ORDER}
    for name,entry in result['directions'].items():
        p = OUTPUT/name
        assert sha(p/'SEAL.json')==entry['seal']
        stamp = read(p/'SEAL.json')
        assert stamp['contract']==result['contract'] and stamp['n']==entry['n']
        assert not stamp['target_box_read'] and not stamp['FINAL_access']
        assert all(sha(p/x)==digest for x,digest in stamp['files'].items())
        assert sha(p/'TIMING.json')==stamp['timing']
    return result


def main():
    contract = lock()
    state('frozen_combinations','Replay the locked two- and three-round inference combinations.')
    sources = read(PRIVATE/'EXECUTION_LOCK.json')['source_models']
    directions = {}
    for source in PLATFORMS:
        encoder = native_encoder(source)
        model = CombinedDecoder().eval().requires_grad_(False)
        checkpoint = PRIOR/source/'full.pth'
        assert sha(checkpoint)==sources[source]
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model = model.cuda()
        for s,target in ORDER:
            if s==source:
                directions[f'{s}_to_{target}'] = direction(s,target,encoder,model,contract)
        del encoder, model
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(x['n'] for x in directions.values())==N
    json_atomic(OUTPUT/'PREDICTION_SEAL.json',dict(n=N,contract=contract,directions=directions,
                target_box_read=False,target_updates=0,FINAL_access=False))
    print('R178_SEALED',N,len(ARMS),flush=True)


if __name__ == '__main__':
    main()
