"""Frozen observation-only inference for all arms, with a single sealed output for each input."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from shared_paths import HERE, NEW, OLD, SEG, PRED, SPEC, LEX, ARMS, digest, read, save, archive_npz, inputs, name
from ordered_choice import prepare
from scene_roles import anchor_returns,world,role,marginal_roles,infer
from scene_faces import split_returns,geometry,only_ground,bottom
from physical_evidence import Returns
from joint_coverage import estimate_shape
from roles_joint import affinity
from predict_witness import readout as ordered
from predict_lex import predict as old_lex, ARMS as OLD_ARMS

IMPLEMENTATION=('predict_scene.py','scene_roles.py','scene_faces.py','scene_partition.py','shared_paths.py')


def run_case(z,seg,words,seed,old):
    n=len(ARMS)
    boxes=np.tile(z['raw'],(n,1));raw=boxes.copy()
    refs=np.tile(z['ref'],(n,1));active=np.zeros(n,bool);mass=np.zeros(n)
    k=ARMS.index('previous_noun')
    raw[k],refs[k],boxes[k],active[k]=old
    x=seg['xyz'].astype(float);split=split_returns(x)
    k=ARMS.index('v1_ground');floor=bottom(raw[0],x,seg['plane'],split)
    boxes[k]=only_ground(raw[0],floor)
    T=np.empty((0,7));margins=np.zeros((n,0));comps=np.zeros((8,2),int)
    pair_mass=np.zeros(n)
    def packet():
        assert np.isfinite(boxes).all() and np.all(boxes[:,3:6]>0)
        return dict(raw=raw,boxes=boxes,ref=refs,applied=active,activation_mass=mass,
            candidates=T,subject_scores=margins,components=comps,pair_mass=pair_mass)
    if not z['active'] or min(words)<0:return packet()
    T,_,mt,_=prepare(z,'t')
    R,_,mr,_=prepare(z,'r')
    margins=np.zeros((n,len(T)))
    if not len(T) or not len(R):return packet()
    points=Returns(x);np.testing.assert_array_equal(points.xyz,x)
    tr=anchor_returns(T,points);rr=anchor_returns(R,points)
    tables=[];labels=seg['labels']
    for level in range(8):
        s=world(seg,level,seed)
        t,r=role(tr,mt,words[0],s),role(rr,mr,words[1],s)
        tables.append((t,r))
        comps[level]=[len(t.index),len(r.index)]
    depth=int(z['depth']);side=int(z['side']);immediate=bool(z['immediate'])
    k=ARMS.index('independent')
    st,sr=marginal_roles(tables,depth,len(T),len(R))
    margins[k]=st;mass[k]=st.sum()
    if st.sum()>0:raw[k]=boxes[k]=T[st.argmax()];active[k]=True
    if sr.sum()>0:refs[k]=R[sr.argmax()]
    for arm in ('separate_worlds','separate_scales','wrong_reference','image_center','joint_raw'):
        k=ARMS.index(arm)
        joint,parts=infer(tables,labels,side,depth,immediate,arm,len(T),len(R))
        margins[k]=joint.sum(1)
        mass[k]=joint.sum()
        if mass[k]<=0:continue
        i=int(margins[k].argmax());j=int(joint[i].argmax())
        raw[k]=boxes[k]=T[i];refs[k]=R[j];active[k]=True;pair_mass[k]=joint[i,j]
    jraw=ARMS.index('joint_raw')
    if not active[jraw]:return packet()
    i=int(margins[jraw].argmax());j=int(joint[i].argmax());anchor=T[i]
    # The raw endpoint, its total evidence and the conditioned pair are identical in each geometry arm.
    floor=bottom(anchor,x,seg['plane'],split)
    for mode in ('fused_boxes','ground_only','marginal_shape','joint_shape'):
        k=ARMS.index(mode)
        raw[k]=anchor;refs[k]=R[j];active[k]=True;mass[k]=mass[jraw]
        margins[k]=margins[jraw];pair_mass[k]=joint[i,j]
        if mode=='fused_boxes':
            a=z['a']; weights=z['ma']*affinity(anchor[None],a)[0]
            boxes[k]=estimate_shape(np.vstack((anchor,a)),np.r_[max(mt[i],0.),weights],anchor)
        elif mode=='ground_only':boxes[k]=only_ground(anchor,floor)
        else:boxes[k]=geometry(anchor,x,split,labels,parts,i,j,floor,mode=='joint_shape')
    return packet()


def worker(task):
    row,zpath,zsha,lpath,lsha,fid,words,oldpath,oldsha,gsha=task
    assert digest(zpath)==zsha and digest(lpath)==lsha and digest(SEG/(fid+'.npz'))==gsha
    with np.load(zpath) as src,np.load(SEG/(fid+'.npz')) as gg:
        z={k:src[k] for k in src}
        seg={k:gg[k] for k in gg}
        if oldpath:
            assert digest(oldpath)==oldsha
            with np.load(oldpath) as prev:
                k=OLD_ARMS.index('joint_raw')
                old=tuple(prev[v][k] for v in ('raw','ref','boxes','applied'))
        else:
            with np.load(lpath) as h:
                prev=old_lex(z,ordered(z),h,words)
            k=OLD_ARMS.index('joint_raw')
            old=tuple(prev[v][k] for v in ('raw','ref','boxes','applied'))
        data=run_case(z,seg,words,int(fid[:8],16),old)
    dst=PRED/name(row)
    archive_npz(dst,**data)
    return dst.name,digest(dst)


def main():
    pins={}
    for g in ('old','new'):
        d=read(SEG/(g+'.json'))
        assert d['protocol']==digest(SPEC) and d['code']==digest(HERE/'scene_partition.py')
        assert not (pins.keys()&d['files'].keys())
        pins.update(d['files'])
    roster=list(inputs())
    PRED.mkdir(exist_ok=True,mode=0o700)
    tasks=[(*item,pins[item[5]]) for item in roster]
    files={}
    with ProcessPoolExecutor(max_workers=8) as jobs:
        for i,(n,h) in enumerate(jobs.map(worker,tasks,chunksize=2),1):
            files[n]=h
            if i%100==0 or i==len(tasks):print('predictions',i,len(tasks),flush=True)
    assert len(files)==len(roster) and all(digest(PRED/n)==v for n,v in files.items())
    save(PRED/'SEALED.json',dict(n=len(files),files=files,protocol=digest(SPEC),
         partition={s:digest(SEG/(s+'.json')) for s in ('old','new')},new=digest(NEW/'SEALED.json'),
         old=digest(LEX/'private/observations/SEALED.json'),code={s:digest(HERE/s) for s in IMPLEMENTATION},
         labels_used=False,source_training=0,target_updates=0,FINAL_access=False))


if __name__=='__main__':main()
