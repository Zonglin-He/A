"""Target-frozen, same-observation six-direction inference. No task labels are loaded."""
import gc
import time

import numpy as np
import torch

from scope175 import (CELLS, CONTROL, HERE, N, ORDER, PLATFORMS, PRIVATE, R165_DATA,
                      SPEC, array_sha, check_locked, json_atomic, npz_atomic, ordinary_rows,
                      points_for, read, sha, state)
from evidence import encoder,extract
from features175 import plane
from model175 import RelationalDecoder
from predict import from_block
from roles175 import roles_batch
from train175 import TRAIN_PIN
from crossplatform.ground_single_frame import orient_box

PRED_ARMS=(*CELLS,*CONTROL)
OUTPUT=PRIVATE/'predictions'
EVAL_PIN=('scope175.py','roles175.py','source175.py','features175.py','model175.py',
          'train175.py','predict175.py','score175.py','audit175.py','mechanism175.py','protocol.json')
BATCH=16


def eval_lock():
    check_locked()
    weights={}
    for s in PLATFORMS:
        p=PRIVATE/s
        arm=read(p/'MODEL_SEAL.json')
        assert arm['source']==s and arm['source_only_selection'] and not arm['target_labels_read']
        for c in CELLS:
            record=p/f'{c}_TRAINING.json'
            assert sha(record)==arm['files'][c]
            proof=read(record)
            assert proof['source_only_selection'] and proof['source']==s and proof['cell']==c
            assert all(proof['inputs'][k]==sha(HERE/k) for k in TRAIN_PIN)
            assert proof['selected_epoch'] in read(SPEC)['training']['checkpoint_epochs']
            weights[f'{s}/{c}']=sha(p/f'{c}.pth')
            assert weights[f'{s}/{c}']==proof['checkpoint']
    value=dict(code={x:sha(HERE/x) for x in EVAL_PIN},models=weights,ordinary_n=N,
               input=sha(PRIVATE/'INPUT_LOCK.json'),predicted_arms=list(PRED_ARMS),
               FINAL_access=False,target_labels_read=False)
    p=PRIVATE/'EVALUATION_LOCK.json'
    if p.exists():
        assert read(p)==value
    else:
        json_atomic(p,value)
    return sha(p)


def models_for(source):
    found={}
    for c in CELLS:
        model=RelationalDecoder(c)
        model.load_state_dict(torch.load(PRIVATE/source/f'{c}.pth',weights_only=True,map_location='cpu'))
        found[c]=model.cuda().eval().requires_grad_(False)
    return found


def evidence_batch(source,target,bank,loaded,lo,hi,check_coordinates=False):
    rows=ordinary_rows(target)
    chunk=[rows[int(j)] for j in bank['ordinals'][lo:hi]]
    joined=[points_for(row) for row in chunk]
    for i,(points,ids) in enumerate(joined,lo):
        assert array_sha(points)==bank['points_sha'][i].tobytes()
        assert array_sha(ids)==bank['indices_sha'][i].tobytes()
    sampled=np.stack([a for a,_ in joined])
    archive=from_block(bank,lo,hi)
    context=roles_batch(chunk)
    f=extract(loaded,[dict(row,spans=s['spans']) for row,s in zip(chunk,context,strict=True)],
              sampled,archived=archive,coordinate_audit=check_coordinates)
    np.testing.assert_array_equal(f['valid'].sum(-1).cpu().numpy(),np.diff(bank['offsets'][lo:hi+1]))
    relation=np.asarray([s['relation'] for s in context],np.int8)
    f['relation']=torch.as_tensor(relation,device='cuda')
    f['depth_plane']=torch.as_tensor(np.stack([plane(r) for r in chunk]),device='cuda')
    assert bool(((f['relation']!=0)<=f['has_reference']).all())
    # Strictly original raw geometry, preserving the archived candidate indices.
    ids=f['anchors'].cpu().numpy()
    raw=archive[0][np.arange(len(chunk))[:,None,None],ids]
    pca=np.stack([np.stack([orient_box(p,b) for b in boxes.reshape(-1,6)]).reshape(2,4,7)
                  for p,boxes in zip(sampled,raw,strict=True)])
    np.testing.assert_array_equal(pca[...,:3],raw[...,:3])
    np.testing.assert_array_equal(pca[...,3:6],raw[...,3:6].clip(min=1e-6))
    return f,pca


@torch.inference_mode()
def predict_block(source,target,bank,loaded,models,first):
    groups=[]
    for st in range(0,len(bank['ordinals']),BATCH):
        f,candidates=evidence_batch(source,target,bank,loaded,st,min(st+BATCH,len(bank['ordinals'])),
                                     first and st==0)
        rows=np.arange(len(candidates))[:,None]
        arms=[]
        for mode in PRED_ARMS:
            current=mode if mode in CELLS else 'full'
            intervention=mode.removeprefix('full_') if mode in CONTROL else None
            out=models[current](f,intervention=intervention)
            choice=out['choices'].cpu().numpy().astype(np.int8)
            boxes=candidates[rows,np.arange(2)[None],choice]
            selected=out['support']
            torch.testing.assert_close(selected.sum(-1),torch.ones_like(selected[...,0]),rtol=0.,atol=2e-6)
            arms.append(dict(boxes=boxes,choice=choice,support=selected.half().cpu().numpy(),
                             scores=out['passes'][-1][3].cpu().numpy(),
                             field=out['field_strength'].cpu().numpy()))
        assert np.isfinite(candidates).all()
        f0=arms[PRED_ARMS.index('full')]
        unknown=f['relation'].cpu().numpy()==0
        for mode in CONTROL:
            alt=arms[PRED_ARMS.index(mode)]
            np.testing.assert_array_equal(f0['choice'][unknown],alt['choice'][unknown])
            np.testing.assert_array_equal(f0['support'][unknown],alt['support'][unknown])
        values={k:np.stack([g[k] for g in arms],1) for k in arms[0]}
        values.update(slot_boxes=candidates,anchor_ids=f['anchors'].cpu().numpy(),
                      relation=f['relation'].cpu().numpy(),xyz=f['xyz'].cpu().numpy())
        groups.append(values)
    return {k:np.concatenate([p[k] for p in groups]) for k in groups[0]}


def predict_direction(source,target,loaded,models,evaluation):
    route=f'{source}_to_{target}'
    root=R165_DATA/route
    seal=read(root/'SEAL.json')
    anchor=read(root/'BBF_CUDA_RECEIPT.json')
    assert sha(root/'SEAL.json')==anchor['direction_forward_seal']
    n=seal['n']
    assert n==anchor['n']==len(ordinary_rows(target))
    assert not seal['protected_payload_access'] and not seal['oracle_target_class_read']
    out=OUTPUT/route
    out.mkdir(exist_ok=True,parents=True,mode=0o700)
    seen,files=0,{}
    begin=time.perf_counter()
    for name,digest in sorted(seal['files'].items()):
        p=root/name
        assert sha(p)==digest
        dst=out/name
        if not dst.exists():
            with np.load(p,allow_pickle=False) as z:
                bank=dict(z)
            pred=predict_block(source,target,bank,loaded,models,seen==0)
            npz_atomic(dst,**pred,ordinals=bank['ordinals'],arms=np.asarray(PRED_ARMS),
                       lock=evaluation,source_block=digest)
        with np.load(dst,allow_pickle=False) as z:
            assert str(z['lock'])==evaluation and str(z['source_block'])==digest and tuple(z['arms'])==PRED_ARMS
            np.testing.assert_array_equal(z['ordinals'],np.arange(seen,seen+len(z['ordinals'])))
            seen+=len(z['ordinals'])
        files[name]=sha(dst)
        print('R175_FROZEN',route,seen,n,round(time.perf_counter()-begin,1),flush=True)
    assert seen==n
    value=dict(n=n,source=source,target=target,files=files,lock=evaluation,
               archive=sha(root/'SEAL.json'),FINAL_access=False,target_task_label_reads=0,target_updates=0)
    json_atomic(out/'SEAL.json',value)
    return value


def verify():
    locked=eval_lock()
    proof=read(OUTPUT/'PREDICTION_SEAL.json')
    assert proof['n']==N and not proof['FINAL_access'] and not proof['target_task_label_reads']
    assert proof['lock']==locked and len(proof['directions'])==6
    for k,v in proof['directions'].items():
        path=OUTPUT/k/'SEAL.json'
        assert sha(path)==v['seal']
        s=read(path)
        assert s['n']==v['n'] and s['lock']==locked and not s['FINAL_access']
        assert all(sha(path.parent/q)==h for q,h in s['files'].items())
    return proof


def main():
    locked=eval_lock()
    state('frozen_prediction','Selected only from source monitors. Run six ordinary directions now.')
    all_proofs={}
    for source in PLATFORMS:
        loaded=encoder(source)
        models=models_for(source)
        for s,target in ORDER:
            if s!=source:
                continue
            route=f'{s}_to_{target}'
            contract=predict_direction(source,target,loaded,models,locked)
            all_proofs[route]=dict(n=contract['n'],seal=sha(OUTPUT/route/'SEAL.json'))
        del loaded,models
        gc.collect()
        torch.cuda.empty_cache()
    assert sum(q['n'] for q in all_proofs.values())==N
    json_atomic(OUTPUT/'PREDICTION_SEAL.json',dict(n=N,directions=all_proofs,lock=locked,
                                     FINAL_access=False,target_task_label_reads=0))
    verify()
    print('R175_PREDICTION_SEALED',N,flush=True)


if __name__=='__main__':
    main()
