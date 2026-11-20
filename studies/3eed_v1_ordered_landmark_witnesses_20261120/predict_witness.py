"""Perform the same frozen direct and controlled readouts, with annotation-free inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from paths_witness import HERE, WORK, OBS, PRED, SPEC, name, read, save, digest
from collect import archive_npz
from roles_joint import affinity
from joint_coverage import estimate_shape
from ordered_choice import prepare, relations, reference_mass, joint, choices
from chord_shape import chords, estimate, blend_views

ARMS=('v1','independent','side','single_cut','misbound','witness_raw','marginal_chord','simple_mean','witness_chord')

def readout(z):
    raw=np.tile(z['raw'],(len(ARMS),1)); refs=np.tile(z['ref'],(len(ARMS),1));boxes=raw.copy()
    ok=np.zeros(len(ARMS),bool);evidence=np.zeros(len(ARMS));info=dict(raw=raw,ref=refs,boxes=boxes,applied=ok,evidence=evidence)
    if not z['active']:return info
    T,pt,mt,tp=prepare(z,'t'); R,pr,mr,rp=prepare(z,'r')
    if min(len(T),len(R))==0:return info
    dist=relations(tp,rp,int(z['side']))
    kt=affinity(T,T);kr=affinity(R,R)
    depth=int(z['depth']);immediate=bool(z['immediate'])
    ref=reference_mass(pr,mr,rp,depth,False,None)
    wrong=reference_mass(pr,mr,rp,depth,True,None)
    hard=reference_mass(pr,mr,rp,depth,False,.5)
    J=joint(pt,ref,dist,immediate)
    for k,p in enumerate((np.outer(mt,ref),joint(pt,ref,dist,False),joint(pt,hard,dist,immediate,.5),
                          joint(pt,wrong,dist,immediate),J),1):
        pick=choices(p,kt,kr)
        evidence[k]=p.sum()
        if pick is None:continue
        i,j=pick;raw[k]=T[i];refs[k]=R[j];boxes[k]=raw[k];ok[k]=True
    if not ok[5]:return info
    i,j=choices(J,kt,kr);anchor=T[i]
    a=z['a'];ma=z['ma']
    order=np.lexsort((ma,)+tuple(a[:,j] for j in range(6,-1,-1)))
    a,ma=a[order],ma[order]
    # Only full-expression geometries overlapping the anchor can contribute to its shape.
    hit=affinity(anchor[None],a)[0]>0
    f=a[hit]
    pool=np.vstack((T,f));ints,inside=chords(anchor,pool)
    transfer=affinity(f,T) if len(f) else np.empty((0,len(T)))
    weights=[]
    for belief in (J.sum(1),J@kr[j]):
        w=kt[i]*belief
        weights.append(blend_views(w,ma[hit]*(transfer@w),inside,len(T)))
    for k in (6,7,8):
        raw[k]=anchor;refs[k]=R[j];evidence[k]=J.sum();ok[k]=True
    boxes[6]=estimate(anchor,ints,weights[0])
    boxes[7]=estimate_shape(pool,weights[1],anchor)
    boxes[8]=estimate(anchor,ints,weights[1])
    return info

def job(item):
    n,checksum=item
    assert digest(OBS/n)==checksum
    with np.load(OBS/n,allow_pickle=False) as f:
        z={k:f[k] for k in f.files}
    result=readout(z);dst=PRED/n;archive_npz(dst,**result)
    return n,digest(dst)

def run(workers):
    o=read(OBS/'SEALED.json');protocol=read(SPEC)
    assert o['pin']['protocol']==digest(SPEC) and list(ARMS)==protocol['arms']
    assert o['pin']['roster']==digest(WORK/'roster.json') and o['n']==len(o['files'])
    PRED.mkdir(mode=0o700,exist_ok=True);files={}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i,(n,h) in enumerate(ex.map(job,o['files'].items(),chunksize=3),1):
            files[n]=h
            if i%100==0:print(i,o['n'],flush=True)
    assert len(files)==o['n'] and all(digest(PRED/n)==h for n,h in files.items())
    save(PRED/'SEALED.json',dict(files=files,protocol=digest(SPEC),obs=digest(OBS/'SEALED.json'),n=len(files),
        code={k:digest(HERE/k) for k in ('predict_witness.py','ordered_choice.py','chord_shape.py')},
        labels_used=False,target_updates=0,source_training=0,FINAL_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);run(p.parse_args().workers)
