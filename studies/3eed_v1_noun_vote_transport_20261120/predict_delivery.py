"""Run every arm on the same measured inventory; archive decisions before target scoring."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from pathlib import Path
import numpy as np
from vote_paths import *
from delivery import evidence,fit_faces,build
from role_cut import structure,layout,shared,medoid
from physical_nodes import assemble,overlap_prior
from physical_evidence import Returns
from scene_partition import build as partition
from terrain_followup import boxes_in_model
from scope183 import VIEWS
from verify_delivery import check

CODE=('vote_paths.py','observe_votes.py','return_votes.cpp','delivery.py','predict_delivery.py','verify_delivery.py')
def codepins():
    return {k:digest(HERE/k) for k in CODE}

def to_model(row,stored):
    if row['scope']=='panel':
        return {k:stored[n] for k,n in [('raw','raw'),('anchored','final'),('ref','ref'),('ref_anchored','ref_final')]}
    view=int(stored['view'])
    return {k:boxes_in_model(stored[n],row['target'],view) for k,n in [('raw','raw'),('anchored','completed'),
                                                            ('ref','reference'),('ref_anchored','reference_completed')]}

def compare_pose(b,c):
    assert np.array_equal(np.isfinite(b),np.isfinite(c))
    if not np.isfinite(b).all():return 0.
    d=b-c
    d[6]=(d[6]+np.pi/2)%np.pi-np.pi/2
    return float(abs(d).max())

def choices(data,m,K,fields,us,ur,words,side,depth,nearest):
    valid=(K==0);view=data['view'];visible=np.isfinite(view[:,0])
    selection={}
    ordered=[arm for arm in ARMS[1:] if arm!='independent']+['independent']
    for arm in ordered:
        if arm in ('independent','misbound_reference'):
            p=fields['association']
        else:p=fields[arm]
        if arm=='independent':
            j,rf=cache
            si=K@(us*m*p[:,words[0]]*visible);ri=K@(ur*rf)
            i=int(np.argmax(np.where(visible,si,-1))) if si.max(initial=0)>0 else -1
            r=int(np.argmax(np.where(visible,ri,-1))) if ri.max(initial=0)>0 else -1
        else:
            mode='wrong_reference_binding' if arm=='misbound_reference' else 'physical'
            j,rf=shared(data['parent'],valid,data['stop'],p,words,view,side,depth,nearest,mode)
            if arm=='association':cache=(j,rf)
            i,r,_=medoid(j,K,us,ur,valid)
        selection[arm]=(i,r)
    return selection

def job(task):
    fid,info,rows,vfile,vhash=task
    start=perf_counter();fields=None;stats={}
    if info:
        assert digest(info['lex'])==info['lh'] and digest(info['word'])==info['wh'] and digest(vfile)==vhash
        with np.load(info['lex']) as lex,np.load(info['word']) as word,np.load(vfile) as observed:
            if info['tree']:
                file,pin=info['tree'];assert digest(file)==pin
                with np.load(file) as tree:data=dict(tree)
            else:
                data=assemble(partition(lex['sample'],lex['logits'],lex['xyz'],info['f']),word)
            np.testing.assert_equal(data['xyz'],np.unique(lex['sample'],axis=0))
            np.testing.assert_equal(lex['xyz'],word['xyz'])
            fields,aux,K,m,stats=evidence(data,lex,observed['center'],word['extent'])
            valid_full=(aux[1][:,:,3:]>1.e-6).all(-1)
            distance=np.linalg.norm(observed['center']-observed['center'].mean(axis=1,keepdims=True),axis=-1)
            stats.update(nodes=len(m),word_center_rms=float(np.sqrt(np.mean(distance**2))),positive_extents=int(valid_full.sum()))
        obs=Returns(data['xyz']);faces={}
        def face(i,c,neutral):
            if (i,c,neutral) not in faces:faces[i,c,neutral]=fit_faces(data,aux,i,c,neutral)
            return faces[i,c,neutral]
        scene_file=WORK/'fields'/(fid+'.npz')
        archive_npz(scene_file,**fields,boxes=data['boxes'],completed=data['completed'],visible=np.isfinite(data['view'][:,0]),m=m)
        fieldpin=digest(scene_file)
    else:fieldpin=None
    results={};maxerr=0.
    for row in rows:
        prior=row['prior'];rp=row['role']
        assert digest(prior)==row['ph'] and digest(rp)==row['rh']
        with np.load(prior) as past,np.load(rp) as z:
            oldplanes=to_model(row,past)
            out={k:np.tile(v[0],(len(ARMS),1)) for k,v in oldplanes.items()}
            for k,from_k in [('fit','anchored'),('ref_fit','ref_anchored'),('null_fit','anchored'),('ref_null_fit','ref_anchored')]:
                out[k]=out[from_k].copy()
            idx=np.full((len(ARMS),2),-1,int)
            if row['active'] and fields is not None and len(data['parent']):
                words=row['words']; us=overlap_prior(data,z,'t',obs);ur=overlap_prior(data,z,'r',obs)
                res=choices(data,m,K,fields,us,ur,words,int(z['side']),int(z['depth']),bool(z['immediate']))
                for arm,(i,r) in res.items():
                    j=ARMS.index(arm)
                    for role,node,c,pre in [(0,i,words[0],''),(1,r,words[1],'ref_')]:
                        if node<0:continue
                        idx[j,role]=node
                        out['raw' if role==0 else 'ref'][j]=data['boxes'][node]
                        out[pre+'anchored'][j]=data['completed'][node,c]
                        out[pre+'fit'][j]=face(node,c,False)
                        out[pre+'null_fit'][j]=face(node,c,True)
            j=ARMS.index('tree');orig=old.ARMS.index('shared_cut')
            maxerr=max([maxerr]+[compare_pose(out[k][j],a[orig]) for k,a in oldplanes.items()])
            assert maxerr<1.e-5,(row['scope'],maxerr)
            fn=WORK/'predictions'/(row_id(row)+'.npz')
            archive_npz(fn,**out,selected=idx)
            results[row_id(row)]=digest(fn)
    return fid,results,fieldpin,stats,maxerr,perf_counter()-start

def run(src,workers):
    cat=catalog();build();checks,err=check()
    votes=WORK/'votes'/src
    cap=read(votes/'SEALED.json')
    assert cap['pins']['catalog']==digest(WORK/'CATALOG.json') and cap['pins']['code']==digest(HERE/'observe_votes.py')
    targets=defaultdict(list)
    for row in cat['rows']:
        if row['source']==src:targets[row['frame']].append(row)
    active={i:v for i,v in cat['frames'].items() if v['source']==src}
    assert set(active)==set(cap['files']) and set(active)<=set(targets)
    for folder in ('fields','predictions'):(WORK/folder).mkdir(mode=0o700,exist_ok=True)
    tasks=[(k,active.get(k),rows,votes/(k+'.npz'),cap['files'].get(k)) for k,rows in targets.items()]
    files={};observed={};stats=[];maxerr=0.;t=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,(fid,out,pin,desc,delta,seconds) in enumerate(pool.map(job,tasks,chunksize=1),1):
            assert files.keys().isdisjoint(out)
            files.update(out);maxerr=max(maxerr,delta)
            if pin:
                observed[fid]=pin;stats.append(dict(frame_source=fid,source=src,**desc))
            if i%150==0:print(src,'frames',i,len(tasks),'predictions',len(files),'seconds',round(perf_counter()-t),flush=True)
    save(WORK/(src+'_activity.json'),stats)
    save(WORK/(src+'_out.json'),dict(source=src,files=files,fields=observed,code=codepins(),pins=pins(),capture=digest(votes/'SEALED.json'),
        checks=checks,numeric_error=err,prior_reproduction_error=maxerr,activity=digest(WORK/(src+'_activity.json')),
        seconds=perf_counter()-t,labels_used=False))

def seal():
    cat=catalog();allrows={};fields={};sources={}
    for src in SOURCES:
        file=WORK/(src+'_out.json');rec=read(file)
        assert rec['code']==codepins() and rec['pins']==pins() and not rec['labels_used']
        assert rec['capture']==digest(WORK/'votes'/src/'SEALED.json') and rec['activity']==digest(WORK/(src+'_activity.json'))
        assert not (allrows.keys()&rec['files'].keys()) and not(fields.keys()&rec['fields'].keys())
        allrows.update(rec['files']);fields.update(rec['fields']);sources[src]=digest(file)
    assert set(allrows)=={row_id(r) for r in cat['rows']} and set(fields)==set(cat['frames'])
    for k,h in allrows.items():assert digest(WORK/'predictions'/(k+'.npz'))==h
    for k,h in fields.items():assert digest(WORK/'fields'/(k+'.npz'))==h
    save(WORK/'SEALED.json',dict(observed=fields,files=allrows,sources=sources,protocol=digest(SPEC),
        catalog=digest(WORK/'CATALOG.json'),code=codepins(),original_full_predictions=pins()['ordinary_seal'],
        total_ordinary=18620,eligible_ordinary=1294,other_scopes=1332,labels_used=False,FINAL_access=False))
    print('predictions and untouched ordinary fallback sealed',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=(*SOURCES,'seal'));p.add_argument('--workers',type=int,default=5)
    a=p.parse_args()
    if a.stage=='seal':seal()
    else:run(a.stage,a.workers)

