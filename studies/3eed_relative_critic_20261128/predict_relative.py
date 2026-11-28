"""Apply the sealed rerankers and the three independent ViT interventions."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from source_geometry import HERE,PRIOR,RUN,digest,read,write
from feature_bank import base_index,describe
from competitive import rebind,silhouettes
from physical_readout import lattice
from prepare_blind import sealed_inputs
from train_relative import logits,model_for


ARMS=('Minimal-D','old_vit_select','competitive_2a','competitive_2b','pool_aabb','pool_vit_old',
      'pool_vit_compact','pool_vit_wide','sim_iou','real_iou','real_pair','real_delta0','real_delta_val',
      'joint_delta0','joint_delta_val')
RANKS=('real_iou','real_pair','real_delta','joint_delta')
CACHE={}


def select(values,tie):
    best=np.flatnonzero(values==np.max(values))
    if tie is not None and tie in best:
        return tie
    return int(best[0])


def select_preferred(score,reference,threshold):
    k=int(np.argmax(score))
    return k if score[k]>threshold else reference


@torch.inference_mode()
def infer(case):
    ctx=CACHE
    uid=case['uid']; row=ctx['source'][uid]
    component=ctx['visual'][uid]
    old=ctx['pool'][uid]
    assert old['base']==case['base']['box']
    base=old['base']
    boxes=dict.fromkeys(ARMS,base)
    boxes['old_vit_select']=ctx['vit'][uid]['boxes']['vit_select']
    boxes['pool_aabb']=old['boxes']['grid_aabb']
    boxes['sim_iou']=old['boxes']['critic_silhouette']
    active=bool(component['stages'])
    _,compact,_=silhouettes(case,component)
    if compact is not None and len(case['fitted'])==2:
        candidates=case['fitted']
        original=int(case['base']['heading']=='qwen')
        measures=np.asarray([compact.score(b) for b in candidates])
        boxes['competitive_2a']=candidates[select(measures,original)]
    else:
        boxes['competitive_2a']=boxes['old_vit_select']
    if active:
        boxes['competitive_2b']=rebind(row['frame'],row['raw'],component['stages'])['box']
    result=dict(uid=uid,target=case['target'],boxes=boxes,active=active,compact=compact is not None,
                role=case['base']['readout'])
    pool=old.get('pool')
    if not pool:
        return result
    # Check the original target lattice byte for byte. No additional target proposals.
    np.testing.assert_array_equal(np.asarray(pool),lattice(case))
    pivot=base_index(pool,base)
    descriptors=describe(case,pool,component)
    assert np.isfinite(descriptors).all()
    values={
        'pool_vit_old':descriptors[:,47],
        'pool_vit_compact':np.array([compact.score(b) for b in pool]) if compact else descriptors[:,47],
        'pool_vit_wide':descriptors[:,48],
        'sim_iou':np.asarray(old['detail']['energy_silhouette']),
    }
    selected={}
    weights=torch.tensor(descriptors)[None]
    anchor=torch.tensor([0 if pivot is None else pivot],dtype=torch.long)
    for name in RANKS:
        if pivot is None and name.endswith('delta'):
            continue
        prediction=logits(ctx['net'][name],name,weights,anchor)[0].numpy()
        if name.endswith('delta'):
            assert abs(float(prediction[pivot]))<1.e-7
        values[name]=prediction
    for name in ('pool_vit_old','pool_vit_compact','pool_vit_wide','real_iou','real_pair'):
        selected[name]=select(values[name],pivot)
    for name in ('real_delta','joint_delta'):
        if pivot is not None:
            for tag,threshold in (('0',0.),('_val',ctx['threshold'][name])):
                selected[name+tag]=select_preferred(values[name],pivot,threshold)
    for name,idx in selected.items():
        boxes[name]=pool[idx]
    result.update(scored_pool=True,selected=selected,reference=pivot,
                  rankings={a:np.asarray(v,dtype=float).tolist() for a,v in values.items()})
    return result


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    template=RUN/'source'/'training-seal.json'
    training=read(template)
    assert not training['used_target'] and training['code']==digest(HERE/'train_relative.py')
    source,*_=sealed_inputs()
    previous=PRIOR/'inputs-private.json'
    assert digest(previous)==read(PRIOR/'input-seal.json')['evidence']
    cases=read(previous)
    now=RUN/'target'
    vis=now/'visual-private.json'
    visual_seal=read(now/'visual-seal.json')
    assert not visual_seal['labels_read'] and visual_seal['train']==digest(template)
    assert visual_seal['visual_sha']==digest(vis) and visual_seal['source_inputs']==digest(previous)
    pool=PRIOR/'critic-blind-private.json'
    vit=PRIOR/'vit-blind-private.json'
    for name,resource in (('critic',pool),('vit',vit)):
        assert digest(resource)==read(PRIOR/f'{name}-prediction-seal.json')['payload']
    nets={}
    for name,meta in training['models'].items():
        f=RUN/'source'/f'{name}.safetensors'
        assert meta['sha']==digest(f)
        net=model_for(name).eval().requires_grad_(False)
        net.load_state_dict(load_file(str(f)))
        nets[name]=net
    CACHE.update(visual=read(vis),pool={a['uid']:a for a in read(pool)},vit={a['uid']:a for a in read(vit)},
                 net=nets,source=source,threshold={k:v['source_val']['threshold'] for k,v in training['models'].items()})
    start=perf_counter()
    with ProcessPoolExecutor(8) as workers:
        records=[]
        for row in workers.map(infer,cases,chunksize=3):
            records.append(row)
            if len(records)%150==0 or len(records)==len(cases):
                print('Target paired predictions',len(records),'/',len(cases),flush=True)
    assert len(records)==873
    out=now/'boxes-private.json'
    write(out,records)
    write(now/'prediction-seal.json',dict(payload=digest(out),n=len(records),counts=dict(Counter(r['target'] for r in records)),
          visual=digest(now/'visual-seal.json'),training=digest(template),prior_candidate_pool=digest(pool),
          labels_read=False,seconds=perf_counter()-start,arms=ARMS,code=digest(__file__)))


if __name__=='__main__':
    main()
