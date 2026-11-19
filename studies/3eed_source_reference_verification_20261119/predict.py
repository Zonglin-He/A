"""Capture R171 D donors and independent reference phrase scores without identity labels."""
import fcntl
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from review import confirmed_prep
from scope import HERE, ROOT, PLATFORMS, PRIVATE, SPEC, PRIOR, atomic_json, read, sha


def points_in(boxes,points):
    from geometry162 import in_box

    return np.stack([in_box(points,b) for b in boxes]) if len(boxes) else np.empty((0,len(points)),bool)


def project_probabilities(scores,native,physical,points):
    from geometry162 import in_box

    source=points_in(np.c_[native,np.zeros(len(native))],points).astype(np.float32)
    target=points_in(physical,points).astype(np.float32)
    intersection=target @ source.T
    union=target.sum(1)[:,None]+source.sum(1)[None]-intersection
    j=np.divide(intersection,union,out=np.zeros_like(union),where=union>0)
    w=np.where((intersection>=3)&(j>=.25),j,0.).astype(float)
    with np.errstate(divide='ignore'):
        scalar=logsumexp(scores+np.log(w),axis=1)-np.log(np.maximum(w.sum(1),np.finfo(float).tiny))
    assert np.array_equal(np.isfinite(scalar),(w>0).any(1))
    return scalar,np.max(w,axis=1)


def distinctive(c,n,support):
    """Return bounded source-only features and preserve a neutral unavailable slot."""
    c_present=np.isfinite(c)
    both = c_present & np.isfinite(n)
    c_rank=np.zeros(len(c))
    lift=np.zeros(len(c))
    if c_present.any():
        c_rank[c_present]=np.clip(c[c_present]-c[c_present].max(),-2.,0.)
    if both.any():
        contrasts=c[both]-n[both]
        lift[both]=np.clip(contrasts-np.median(contrasts),-2.,2.)
    return np.column_stack((c_rank,lift,support))


def retrieve(source, row, views, name):
    from common160 import array_sha
    from crossplatform.ground_single_frame import orient_box
    from predict165 import points_for
    from prepare76 import geometry
    from requery169 import native_donors, native_forward
    from assignment171 import reference_unary

    start=perf_counter()
    pts, ix=points_for(row)
    points=np.unique(pts,axis=0)
    d=native_forward(source,row,pts,ix,None,views['D'])
    z=d['z']
    np.testing.assert_array_equal(z['point_indices'],ix)
    ids, ranks, _=native_donors(d)
    assert len(ids) and ids[0]==d['subject_winner']
    donor=z['boxes'][ids]
    boxes=np.stack([orient_box(pts,box) for box in donor])
    unaries=reference_unary(ranks)
    np.testing.assert_array_equal(unaries[:-1],ranks-ranks.max())
    assert unaries.argmax()==0
    fields,_=geometry(dict(row,caption=views['D']['text']),z)
    order=np.sort(np.unique(z['xyz'][0],axis=0,return_index=True)[1])
    xyz=z['xyz'][0][order]
    uv=fields['uv_grid'][order]/[48,32]
    visible=fields['visible'][order]
    values=[]
    for arm in ('C','noun'):
        x=native_forward(source,row,pts,ix,None,views[arm])
        np.testing.assert_array_equal(x['z']['point_indices'],ix)
        np.testing.assert_array_equal(x['z']['xyz'],z['xyz'])
        a,w=project_probabilities(np.log(x['probability']),x['z']['boxes'],
                                  np.c_[donor,np.zeros(len(donor))],points)
        values.append((a,w))
    (c,cs),(noun,ns)=values
    design=distinctive(c,noun,np.minimum(cs,ns))
    nuisance=design.copy()
    if len(ids)>1:
        value=hashlib.sha256(('reference-wrong-match-v1|'+name).encode()).digest()
        shift=1+int.from_bytes(value[:8],'little')%(len(ids)-1)
        nuisance=np.roll(nuisance,shift,axis=0)
    route_prob=d['probability']
    delta=float(unaries[0]-unaries[1]) if len(ids)>1 else 0.
    reliability=np.array((float(route_prob[ids[0]]),np.clip(delta,0.,8.),len(ids)/32))
    return dict(boxes=boxes,raw=donor.astype(float),points=points,seeds=xyz,uv=uv,visible=visible,
                sample_sha=array_sha(ix),points_sha=array_sha(pts),unaries=unaries,c=c,noun=noun,c_overlap=cs,n_overlap=ns,
                evidence=design,mismatched=nuisance,reliability=reliability,elapsed=perf_counter()-start,
                frozen_query_count=3)


def predict():
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz

    out=PRIVATE/'predictions'
    out.mkdir(parents=True,mode=0o700,exist_ok=True)
    certificate=out/'SEAL.json'
    assert not certificate.exists()
    by_source={s:[(item['case'], {k:item['row'][k] for k in ('id','seed','caption','platform','suffix','imagepath',
                                      'metapath','pointpath','point_sha256','image_sha256')}, item['views'])
                     for item in confirmed_prep(s)] for s in PLATFORMS}
    intended=dict(protocol=sha(SPEC),program=sha(Path(__file__)),
                  inputs={s:sha(PRIVATE/'prep'/f'{s}.json') for s in PLATFORMS},
                  base={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'crossplatform/single_frame_sources.json',
                       ROOT/'research/round169/requery169.py', ROOT/'research/round171/assignment171.py',
                       ROOT/'research/round76/prepare76.py')},
                  truth_opened=False,protected_data_access=False)
    lock=out/'LOCK.json'
    if lock.exists():
        assert read(lock)==intended
    else:
        atomic_json(lock,intended)
    files={}
    started=perf_counter()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for s,entries in by_source.items():
            if not entries:
                continue
            model=EvidenceGrounder(s)
            start=fingerprint(model.base),fingerprint(model.matcher)
            for k,(name,row,views) in enumerate(entries,1):
                assert sha(Path(row['pointpath']))==row['point_sha256'] and sha(Path(row['imagepath']))==row['image_sha256']
                path=out/(name+'.npz')
                if path.exists():
                    assert str(read_npz(path)['pin'])==sha(lock)
                else:
                    with torch.inference_mode():
                        item=retrieve(model,row,views,name)
                    npz_atomic(path,**item,pin=sha(lock))
                files[name]=sha(path)
                if k%9==0:
                    print(s,k,'sealed raw source reference records pending completion',flush=True)
            assert start==(fingerprint(model.base),fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert len(files)==sum(map(len,by_source.values()))
    atomic_json(certificate,dict(pin=sha(lock),files=files,source_parameters_unchanged=True,
                   prediction_reads_truth=False,elapsed=perf_counter()-started,protected_data_access=False))
    print('Sealed raw predictions',len(files),flush=True)


if __name__=='__main__':
    predict()
