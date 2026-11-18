"""Contrast the two directions of one predicate while keeping entity phrases fixed."""
from collections import Counter
import argparse
import fcntl
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.special import logsumexp

from binding import transfer_scores
from binding_collect import HERE, ROOT, R171, INPUT, PRIOR_INPUT, PLATFORMS, SPLITS, WORK, atomic_json, read, sha
from edges import point_support

DATA=HERE/'private/polarity'
SPEC=HERE/'polarity_protocol.json'


def opposite(spec):
    pos=spec['predicate_pos']
    orig, repl = ('left','right') if spec['sign']==1 else ('right','left')
    prefix,old,suffix = spec['text'][:pos],spec['text'][pos:pos+len(orig)],spec['text'][pos+len(orig):]
    assert old==orig
    shift=len(repl)-len(old)
    spans=[]
    for a,b in spec['spans']:
        assert b<=pos or a>=pos+len(old)
        spans.append([a+shift,b+shift] if a>=pos+len(old) else [a,b])
    text = prefix+repl+suffix
    assert [text[a:b] for a,b in spans]==[spec['text'][a:b] for a,b in spec['spans']]
    return dict(text=text,spans=spans,predicate_pos=pos,sign=-spec['sign'])


def contrast(both, base, name, role):
    true,other = both
    good = np.isfinite(true)&np.isfinite(other)
    positions = np.flatnonzero(good)
    diff = np.zeros(len(base))
    swapped = diff.copy()
    if len(positions)>=2:
        s = true[good]-logsumexp(true[good])
        t = other[good]-logsumexp(other[good])
        delta = np.clip(s-t,-1.,1.)
        diff[positions] = delta
        token = hashlib.sha256((name+role).encode()).digest()
        turn = 1+int.from_bytes(token[:8],'little')%(len(positions)-1)
        swapped[positions]=np.roll(delta,turn)
    return diff,swapped,good


def collect_features(mod,entry,old,label):
    from common160 import array_sha
    from predict165 import points_for
    from requery169 import native_forward

    sbox,rbox=old['subject_boxes'],old['D_natural_boxes']
    ns,nr=len(sbox),len(rbox)
    values = []
    seconds = 0.
    applied = entry['rewrite']['status']=='supported'
    if applied:
        row=entry['row']
        assert sha(Path(row['pointpath']))==row['point_sha256']
        assert sha(Path(row['imagepath']))==row['image_sha256']
        t=perf_counter()
        pts, ids = points_for(row)
        assert old['point_sha']==array_sha(pts) and old['sample_sha']==array_sha(ids)
        support=np.unique(pts,axis=0)
        for arm,roi in [('B',sbox),('D',rbox)]:
            q=entry['rewrite'][arm]
            for spec in [q,opposite(q)]:
                f=native_forward(mod,row,pts,ids,None,spec)
                native=f['z']['boxes']
                z,_=transfer_scores(np.c_[native,np.zeros(len(native))],roi,support,np.log(f['probability']))
                values.append(z)
        seconds = perf_counter()-t
    else:
        values = [np.full(ns,-np.inf),np.full(ns,-np.inf),np.full(nr,-np.inf),np.full(nr,-np.inf)]
    score_s = old['subject_unary']
    score_r = old['D_native_unary'][:-1]
    delta_s,wrong_s,ok_s=contrast(values[:2],score_s,label,'subject')
    delta_r,wrong_r,ok_r=contrast(values[2:],score_r,label,'reference')
    return dict(plus_s=values[0],minus_s=values[1],plus_r=values[2],minus_r=values[3],delta_s=delta_s,
                delta_r=delta_r,wrong_s=wrong_s,wrong_r=wrong_r,supported_s=ok_s,supported_r=ok_r,
                base_s=score_s,base_r=score_r,eligible=applied,seconds=seconds)


def collect():
    import torch
    from common160 import npz_atomic
    from evidence162 import EvidenceGrounder
    from model51 import fingerprint
    from predict171 import read_npz
    from review171 import verify_predictions

    DATA.mkdir(exist_ok=True,mode=0o700,parents=True)
    lock=DATA/'LOCK.json'
    p=verify_predictions()
    pins=dict(protocol=sha(SPEC),code=sha(Path(__file__)),parent=sha(R171/'PREDICTION_SEAL.json'),
              dependency={n:sha(HERE/n) for n in ('binding.py','edges.py')})
    if lock.exists():
        assert read(lock)==pins
    else:
        atomic_json(lock,pins)
    seal=DATA/'SEAL.json'
    assert not seal.exists()
    catalog=dict(zip(SPLITS,(read(PRIOR_INPUT),read(INPUT)),strict=True))
    files={}
    tally=Counter()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            model=EvidenceGrounder(source)
            before=fingerprint(model.base),fingerprint(model.matcher)
            for cohort,items in catalog.items():
                (DATA/cohort).mkdir(exist_ok=True,mode=0o700)
                for item in items:
                    if source==item['target']:
                        continue
                    label=f"{cohort}/{item['case']:02d}_{source}.npz"
                    prior=WORK/'predictions'/label
                    assert sha(prior)==p['files'][label]
                    target=DATA/label
                    if target.exists():
                        f=read_npz(target)
                        assert str(f['pin'])==sha(lock)
                    else:
                        with torch.inference_mode():
                            f=collect_features(model,item,read_npz(prior),label)
                        npz_atomic(target,**f,pin=sha(lock))
                    files[label]=sha(target)
                    tally[source]+=1
                    if tally[source]%16==0:
                        print(source,tally[source],flush=True)
            assert before==(fingerprint(model.base),fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert files.keys()==p['files'].keys()
    atomic_json(seal,dict(files=files,lock=sha(lock),target_labels_read=False,protected_data_access=False))
    print('Sealed predicate polarity comparisons',len(files),flush=True)


if __name__=='__main__':
    collect()
