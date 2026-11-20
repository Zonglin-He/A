"""Emit decisions and scene membership before loading target labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
from paths_tree import *
from physical_nodes import overlap_prior
from physical_evidence import Returns
from role_cut import decode,build_library
from verify_kernel import run as check_kernel

CODE=('paths_tree.py','physical_nodes.py','role_cut.py','role_cut.cpp','predict_tree.py','verify_kernel.py')

def run_row(task):
    row,role,rhash,imhash,fid,words,sig=task
    n=name(row);start=perf_counter()
    assert digest(SP/n)==imhash and digest(role)==rhash
    with np.load(role) as z,np.load(SP/n) as last:
        count=len(ARMS);baseline=parent.ARMS.index('v1');old=parent.ARMS.index('local_shape')
        raw=np.tile(last['raw'][baseline],(count,1));ref=np.tile(last['ref'][baseline],(count,1))
        after=np.tile(last['final'][baseline],(count,1));ref_after=ref.copy()
        raw[1]=last['raw'][old];ref[1]=last['ref'][old];after[1]=last['final'][old];ref_after[1]=ref[1]
        applied=np.zeros(count,bool);applied[1]=last['applied'][old]
        idx=np.full((count,2),-1,int)
        visids=np.zeros(0,int);boxes=np.zeros((0,7));tbox=boxes;rbox=boxes
        ranks=np.zeros((count,0));nn=0
        if bool(z['active']) and min(words)>=0:
            op=OBJECTS/(fid+'.npz');assert sig and digest(op)==sig
            with np.load(op) as src:data=dict(src)
            nn=len(data['parent'])
            if nn:
                physical=Returns(data['xyz'])
                np.testing.assert_equal(physical.xyz,data['xyz'])
                us=overlap_prior(data,z,'t',physical);ur=overlap_prior(data,z,'r',physical)
                res=decode(data,words,us,ur,int(z['side']),int(z['depth']),bool(z['immediate']))
                visids=np.flatnonzero(np.isfinite(data['view'][:,0]))
                boxes=data['boxes'][visids];tbox=data['completed'][visids,words[0]]
                rbox=data['completed'][visids,words[1]]
                ranks=np.zeros((count,len(visids)))
                for arm,(i,j,m) in res.items():
                    k=ARMS.index(arm);ranks[k]=m[visids]
                    if i>=0:
                        assert i in visids
                        raw[k]=data['boxes'][i];after[k]=data['completed'][i,words[0]]
                        idx[k,0]=i;applied[k]=True
                    if j>=0:
                        assert j in visids
                        ref[k]=data['boxes'][j];ref_after[k]=data['completed'][j,words[1]]
                        idx[k,1]=j
        assert np.isfinite(raw).all() and np.isfinite(after).all()
        np.testing.assert_equal(raw[0],z['raw'])
        archive_npz(PRED/n,raw=raw,final=after,ref=ref,ref_final=ref_after,applied=applied,
            idx=idx,visible_ids=visids,physical_raw=boxes,physical_subject=tbox,physical_ref=rbox,
            scores=ranks,nodes=np.asarray(nn))
    return n,digest(PRED/n),perf_counter()-start,nn

def run(workers):
    pins=base_pins();condition_count,err=check_kernel()
    tree=read(OBJECTS/'SEALED.json')
    assert tree['input']==pins and tree['code']==digest(HERE/'physical_nodes.py') and tree['protocol']==digest(SPEC)
    hist=read(SP/'SEALED.json');rows=observations()
    PRED.mkdir(exist_ok=True,mode=0o700)
    work=[(r,path,rh,hist['files'][name(r)],fid,w,tree['files'].get(fid)) for r,path,rh,_,_,fid,w,*_ in rows]
    files={};times=[];sizes=[];t=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i,(n,h,dt,nn) in enumerate(ex.map(run_row,work,chunksize=1),1):
            files[n]=h;times.append(dt);sizes.append(nn)
            if i%100==0:print('sealed decisions',i,'of',len(work),round(perf_counter()-t,1),'seconds',flush=True)
    assert set(files)=={name(row[0]) for row in rows}
    save(PRED/'SEALED.json',dict(pins=pins,protocol=digest(SPEC),objects=digest(OBJECTS/'SEALED.json'),
         solver_checks=condition_count,solver_max_error=err,kernel_inputs='local build from pinned C++',
         code={n:digest(HERE/n) for n in CODE},files=files,n=len(files),labels_used=False,FINAL_access=False,
         wall_seconds=perf_counter()-t,summed_worker_seconds=sum(times),evaluated_records=sum(n>0 for n in sizes),
         source_updates=0,target_updates=0))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=7);run(p.parse_args().workers)
