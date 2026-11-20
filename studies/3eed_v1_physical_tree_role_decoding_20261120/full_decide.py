"""Freeze full-cohort physical boxes for every readout and both geometry conditions."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
from paths_tree import *
from full_roster import FULL,CFG_FULL,full_pins
from full_observe import doc_pin,key
from scope183 import ordinary,frame
from role_cut import build_library,decode
from scene_partition import build
from physical_nodes import assemble,overlap_prior
from physical_evidence import Returns
from shape_evidence import prepare as lower_bound
from symmetric_shape import two_sided
from scene_roles import world,role,infer,anchor_returns
from ordered_choice import prepare
from scene_faces import split_returns,bottom,only_ground
from terrain_followup import geometry_predictions
from score_roles import sensor_frame_box

def old_read(z,seg,posterior,words,seed):
    T,_,tm,_=prepare(z,'t');R,_,rm,_=prepare(z,'r')
    if not (len(T) and len(R)):return None
    points=Returns(seg['xyz'])
    a=anchor_returns(T,points);b=anchor_returns(R,points)
    local=dict(seg)
    for i in range(8):local['p'+str(i)]=posterior['local'+str(i)]
    tables=[]
    for i in range(8):
        scene=world(local,i,seed);tables.append((role(a,tm,words[0],scene),role(b,rm,words[1],scene)))
    J,_=infer(tables,seg['labels'],int(z['side']),int(z['depth']),bool(z['immediate']),'aligned',len(T),len(R))
    if J.sum()<=0:return None
    i=int(J.sum(1).argmax());return T[i],R[J[i].argmax()]

def case(task):
    source,target,ix,f,words,paths,wordmeta,anchors,history=task
    assert digest(paths['role'])==paths['sha']
    assert digest(wordmeta['lex'])==wordmeta['lex_sha'] and digest(wordmeta['word'])==wordmeta['word_sha']
    with np.load(paths['role']) as h,np.load(wordmeta['lex']) as l,np.load(wordmeta['word']) as word:
        z=dict(h);seg=build(l['sample'],l['logits'],l['xyz'],f)
        data=assemble(seg,word);posterior=two_sided(seg,word,lower_bound(seg,word))
    view=('F','FL','FR','SL','SR').index(f['waymo_view'])
    conv=lambda b:sensor_frame_box(b,target,view)
    raw=np.tile(anchors[0],(len(ARMS),1));after=np.tile(anchors[1],(len(ARMS),1))
    refs=np.tile(conv(z['ref']),(len(ARMS),1)) if np.isfinite(z['ref']).all() else np.full_like(raw,np.nan)
    ref_completed=refs.copy()
    expected=conv(z['raw'])
    delta=float(np.max(abs(expected-raw[0])))
    assert delta<1e-4,(source,target,ix,delta)
    applied=np.zeros(len(ARMS),bool)
    old=old_read(z,seg,posterior,words,int(paths['frame'][:8],16))
    if old is not None:
        raw[1]=conv(old[0]);refs[1]=conv(old[1]);ref_completed[1]=refs[1];applied[1]=True
        x=seg['xyz'].astype(float)
        after[1]=conv(only_ground(old[0],bottom(old[0],x,seg['plane'],split_returns(x))))
    vis=np.isfinite(data['view'][:,0])
    if len(data['parent']):
        pts=Returns(data['xyz'])
        us=overlap_prior(data,z,'t',pts);ur=overlap_prior(data,z,'r',pts)
        choices=decode(data,words[:2],us,ur,int(z['side']),int(z['depth']),bool(z['immediate']))
        for arm,(i,j,_) in choices.items():
            k=ARMS.index(arm)
            if i>=0:
                raw[k]=conv(data['boxes'][i]);after[k]=conv(data['completed'][i,words[0]]);applied[k]=True
            if j>=0:
                refs[k]=conv(data['boxes'][j])
                ref_completed[k]=conv(data['completed'][j,words[1]])
    for k in (0,1):
        if not applied[k]:np.testing.assert_equal(raw[k],anchors[0])
    diff=0.
    if history:
        assert digest(history[0])==history[1]
        with np.load(history[0]) as old:
            for arr,k in [(raw,'raw'),(after,'final')]:
                check=np.asarray([conv(b) for b in old[k]]);diff=max(diff,float(abs(check-arr).max()))
            np.testing.assert_array_equal(old['applied'],applied)
            assert diff<1e-4,(source,target,ix,diff)
    info=dict(raw=raw,completed=after,reference=refs,reference_completed=ref_completed,applied=applied,ordinal=np.array(ix),
              visible_physical=data['boxes'][vis],native_a=z['a'],native_t=z['t_box'],native_r=z['r_box'],
              subject_pool=data['completed'][vis,words[0]],reference_pool=data['completed'][vis,words[1]],
              seeds=z['seeds'],view=np.array(view))
    return ix,info,delta,diff

def compute(source,workers):
    pin=doc_pin()
    observations_file=FULL/'observations'/(source+'.json');observed=read(observations_file)
    assert observed['pin']==pin and observed['protocol']==digest(CFG_FULL) and not observed['labels_used']
    assert observed['code']==digest(HERE/'full_observe.py')
    opt=read(FULL/'parsed.json');table=ordinary()
    terrain={d:z for d,z,_ in geometry_predictions()}
    archived=read(PRED/'SEALED.json')['files']
    previous={}
    for row,*_ in observations():
        f=row['frame']
        if row['source']==source:previous[key(f),f['text'].strip()]=(PRED/name(row),archived[name(row)])
    out=FULL/'predictions';out.mkdir(mode=0o700,exist_ok=True)
    build_library();dirs={};replayed=0;err=0.;common_error=0.;started=perf_counter()
    for target,items in table.items():
        if target==source:continue
        d=source+'_to_'+target;anchors=terrain[d]
        folder=out/d;folder.mkdir(mode=0o700,exist_ok=True)
        registered=np.array([x is not None for x in opt[target]])
        raw=np.repeat(anchors[:,0,None,:],len(ARMS),axis=1)
        finished=np.repeat(anchors[:,1,None,:],len(ARMS),axis=1)
        active=np.zeros(raw.shape[:2],bool)
        tasks=[];files={}
        for i in np.flatnonzero(registered):
            z=observed['records'][target+'_'+str(i)];f=frame(items[i]); old=previous.get((key(f),f['text'].strip()))
            if old:replayed+=1
            tasks.append((source,target,int(i),f,opt[target][i],z,observed['frames'][z['frame']],anchors[i],old))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for k,(i,info,diff,earlier) in enumerate(ex.map(case,tasks,chunksize=1),1):
                raw[i]=info['raw'];finished[i]=info['completed'];active[i]=info['applied']
                p=folder/(str(i)+'.npz');archive_npz(p,**info);files[str(i)]=digest(p)
                err=max(diff,err);common_error=max(common_error,earlier)
                if k%100==0:print(d,k,len(tasks),round(perf_counter()-started),flush=True)
        np.testing.assert_equal(raw[:,0],anchors[:,0]);np.testing.assert_equal(finished[:,0],anchors[:,1])
        assert set(map(int,files))==set(np.flatnonzero(registered)) and not active[~registered].any()
        p=folder/'all.npz';archive_npz(p,measured=raw,completed=finished,applied=active,registered=registered)
        record=folder/'SEALED.json'
        save(record,dict(n=len(items),registered=int(registered.sum()),all=digest(p),files=files))
        dirs[d]=digest(record)
    save(out/(source+'.json'),dict(pins=full_pins(),protocol=digest(CFG_FULL),roster=pin,
         observations=digest(observations_file),old_study=digest(PRED/'SEALED.json'),code=digest(HERE/'full_decide.py'),
         dirs=dirs,replayed=replayed,max_native_error=err,max_small_study_box_error=common_error,training=False,
         task_labels_used=False,FINAL_access=False,seconds=perf_counter()-started))

def seal_all():
    dirs={};obs={};stats={};out=FULL/'predictions'
    pin=doc_pin()
    for src in SOURCES:
        record=out/(src+'.json');v=read(record)
        assert not v['task_labels_used'] and v['pins']==full_pins() and v['protocol']==digest(CFG_FULL)
        assert v['code']==digest(HERE/'full_decide.py') and v['roster']==pin
        assert digest(FULL/'observations'/(src+'.json'))==v['observations']
        obs[src]=digest(record);dirs.update(v['dirs'])
        stats[src]={k:v[k] for k in ('replayed','max_native_error','max_small_study_box_error','seconds')}
    n=0
    for d,h in dirs.items():
        folder=out/d
        assert digest(folder/'SEALED.json')==h
        check=read(folder/'SEALED.json');assert digest(folder/'all.npz')==check['all']
        n+=check['n']
    assert n==18620 and len(dirs)==6
    save(out/'SEALED.json',dict(records=n,dirs=dirs,per_source=obs,small=digest(PRED/'SEALED.json'),
         protocol=digest(CFG_FULL),code=digest(HERE/'full_decide.py'),pins=full_pins(),stats=stats,
         labels_used=False,source_updates=0,target_updates=0,FINAL_access=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=(*SOURCES,'seal'));p.add_argument('--workers',type=int,default=7)
    args=p.parse_args()
    seal_all() if args.stage=='seal' else compute(args.stage,args.workers)
