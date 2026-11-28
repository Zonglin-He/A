"""Generate candidate features with the same operations in source and transfer."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
from collections import Counter
from time import perf_counter

import cv2
import numpy as np
import torch

from source_geometry import HERE,MODEL,PRIOR,RUN,digest,read,rosters,runtime,source_guard,write
from competitive import heads,probability_fields,saved_regions,silhouettes
from geometry_models import critic_features,exact_iou
from physical_readout import lattice,make_scene,rectangle_iou
from prepare_blind import prepare,sealed_inputs
from vision_reads import image,patches,prototype_mask


def source_context(pair):
    import minimal
    row,replies=pair
    frame=row['frame']
    base=minimal.predict(frame,replies)
    wrapped=dict(frame=frame,raw=replies,target='waymo')
    case=prepare((row['uid'],wrapped,dict(outputs=dict(axis_2d=base))))
    regions=saved_regions(frame,replies) if case['first_window'] else {}
    return case,regions


def target_context(pair):
    case,row=pair
    return case,saved_regions(row['frame'],row['raw']) if case['first_window'] else {}


def source_visual(processor,model,cases,frame,*,legacy_cache=None):
    legacy_cache={} if legacy_cache is None else legacy_cache
    result={}
    jobs=[]
    for case,regions in cases:
        uid=case['uid']
        old=legacy_cache.get(uid)
        result[uid]=dict(stages={},legacy=old)
        if (case['first_window'] and ((old is None and bool(case['marks'])) or
                                     any(len(r)>1 for r in regions.values()))):
            jobs.append((case,regions))
    calls=0
    with torch.inference_mode():
        for i in range(0,len(jobs),8):
            batch=jobs[i:i+8]
            feats=patches(processor,model,[image(frame[case['uid']]['rgb']) for case,_ in batch])
            calls+=1
            for (case,regs),feat in zip(batch,feats,strict=True):
                uid=case['uid']; saved=result[uid]
                if saved['legacy'] is None:
                    saved['legacy']=prototype_mask(feat,case['size'],case['window'],case['marks'])
                current='C' if case['base']['readout']=='C_ray' else 'H'
                for name,regions in regs.items():
                    if len(regions)>1:
                        fields=probability_fields(feat,case,regions,matched_primary=name==current)
                        if fields is not None:
                            saved['stages'][name]=fields
            if calls%20==0 or i+8>=len(jobs):
                print('Frozen ViT',min(i+8,len(jobs)),'/',len(jobs),flush=True)
    return result,dict(vision_images=len(jobs),vision_batches=calls,language_decodes=0)


@torch.inference_mode()
def describe(case,pool,visual):
    """Old 48 channels followed by separate primary and rival silhouette overlap."""
    scene=make_scene(case)
    raw=torch.tensor(np.asarray(pool),dtype=torch.float32)
    geo=critic_features(scene.repeat(len(pool)),raw).numpy()
    legacy,compact,wide=silhouettes(case,visual)
    old=np.array([legacy.score(b) for b in pool] if legacy else [0.]*len(pool))
    if wide:
        a=np.asarray([[f.score(b) for b in pool] for f in wide])
        comp=a[0]; rival=a[1:].max(0)
    else:
        comp=old; rival=np.zeros(len(pool))
    return np.column_stack((geo,old,comp,rival)).astype(np.float32)


def base_index(pool,baseline):
    if baseline is None:
        return None
    matches=np.flatnonzero(np.max(abs(np.asarray(pool)-baseline),axis=1)<1.e-7)
    if not len(matches):
        raise ValueError('The fixed candidate pool lost the reference box')
    return int(matches[0])


@torch.inference_mode()
def source_array(cases,visual,truth):
    result=[]
    for case,_ in cases:
        uid=case['uid']
        if case['base']['box'] is None:
            continue
        pool=lattice(case)
        x=describe(case,pool,visual[uid])
        anchor=base_index(pool,case['base']['box'])
        boxes=torch.tensor(pool,dtype=torch.float64)
        truthboxes=torch.tensor(truth[uid],dtype=torch.float64).expand_as(boxes)
        y=exact_iou(boxes,truthboxes).numpy().astype(np.float32)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        result.append((x,y,anchor))
    width=max(len(y) for _,y,_ in result)
    desc=np.zeros((len(result),width,50),np.float32)
    iou=np.zeros((len(result),width),np.float32)
    valid=np.zeros(iou.shape,bool)
    reference=[]
    for i,(x,y,origin) in enumerate(result):
        desc[i,:len(y)]=x;iou[i,:len(y)]=y;valid[i,:len(y)]=True
        reference.append(origin)
    return dict(x=desc,y=iou,mask=valid,base=np.asarray(reference))


def source():
    source_guard()
    runtime()
    from run_full_qwen import load_model
    out=RUN/'source'
    seal=read(out/'decode-seal.json')
    frozen=out/'frozen-source-private.json'
    assert not seal['target_read'] and seal['payload']==digest(frozen)
    replies=read(frozen)
    banks,pins=rosters()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    start=perf_counter()
    observations={}
    with ProcessPoolExecutor(8) as workers:
        for part,train in banks.items():
            data=workers.map(source_context,((r,replies[part][r['uid']]) for r in train),chunksize=3)
            observations[part]=list(data)
            print('Real source candidates',part,len(observations[part]),flush=True)
    processor,model,_=load_model(MODEL)
    infos={}
    for part,train in banks.items():
        frames={r['uid']:r['frame'] for r in train}
        fields,visual=source_visual(processor,model,observations[part],frames)
        target={r['uid']:r['box'] for r in train}
        arr=source_array(observations[part],fields,target)
        f=out/f'features-{part}.npz'
        np.savez(f,**arr)
        infos[part]=dict(sha=digest(f),queries=int(len(arr['y'])),candidate_total=int(arr['mask'].sum()),
                         active_competition=sum(bool(f['stages']) for f in fields.values()),visual=visual)
        print('Sealed source features',part,infos[part],flush=True)
    write(out/'feature-seal.json',dict(info=infos,source=pins,decode=digest(out/'decode-seal.json'),used_target=False,
          seconds=perf_counter()-start,files={p:digest(HERE/p) for p in ('competitive.py','feature_bank.py')}))


def target():
    out=RUN/'target'
    out.mkdir(parents=True,exist_ok=True)
    # Access the target roster only after all model selection has been sealed.
    stamp=read(RUN/'source'/'training-seal.json')
    assert not stamp['used_target'] and len(stamp['models'])==4
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    pinned=read(PRIOR/'input-seal.json')
    bank=PRIOR/'inputs-private.json'
    assert not pinned['labels_read'] and pinned['evidence']==digest(bank)
    cases=read(bank)
    frames,*_=sealed_inputs()
    assert len(frames)==len(cases)==873
    from run_full_qwen import load_model
    with ProcessPoolExecutor(8) as workers:
        records=list(workers.map(target_context,((c,frames[c['uid']]) for c in cases),chunksize=4))
    print('Fixed target regions',len(records),flush=True)
    visual_stamp=read(PRIOR/'vit-seal.json')
    visual_file=PRIOR/'vit-masks-private.json'
    assert visual_stamp['payload']==digest(visual_file)
    old=read(visual_file)
    processor,model,_=load_model(MODEL)
    visual,receipt=source_visual(processor,model,records,{u:r['frame'] for u,r in frames.items()},legacy_cache=old)
    f=out/'visual-private.json'
    write(f,visual)
    # Preserve projected region membership for a separate offline separation test.
    regions=out/'regions-private.json'
    write(regions,{c['uid']:r for c,r in records})
    write(out/'visual-seal.json',dict(visual_sha=digest(f),regions_sha=digest(regions),baseline_cache=visual_stamp['payload'],
          train=digest(RUN/'source'/'training-seal.json'),source_inputs=pinned['evidence'],labels_read=False,
          competitive_queries=sum(bool(v['stages']) for v in visual.values()),**receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('part',choices=('source','target'))
    (source if p.parse_args().part=='source' else target)()
