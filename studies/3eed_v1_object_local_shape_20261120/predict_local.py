"""Seal complete choices for every arm without opening target labels."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
from local_paths import *
from method_local import ARMS,GEOMS,evidence,decide,fit_on_node
from role_cut import structure
from predict_delivery import to_model,compare_pose
from observe_local import signatures

def code():
    return {f:digest(HERE/f) for f in ('local_paths.py','nominate.py','local_reader.py','observe_local.py','method_local.py','predict_local.py','ROLE_CONTROL.json')}

def job(task):
    fid,rows,src,pin,rawpin=task
    plan=None;local=None;result={};err=0.;nomtotal=0
    if pin is not None:
        path=WORK/'plans'/(fid+'.npz');obs=WORK/'observed'/src/(fid+'.npz')
        assert digest(path)==pin and digest(obs)==rawpin
        with np.load(path) as f:plan={k:f[k] for k in ('row','us','ur','nom','boxes','completed','view','stop','parent','size','p','crop_nodes')}
        with np.load(obs) as z:local=dict(z)
        np.testing.assert_equal(local['ids'],plan['crop_nodes'])
        m,K=structure(plan['parent'],plan['stop'],plan['size'])
        lookup={int(node):i for i,node in enumerate(local['ids'])}
        row_lookup={k:i for i,k in enumerate(plan['row'])}
    for row in rows:
        rid=row_id(row);assert digest(row['prior'])==row['ph']
        with np.load(row['prior']) as past:oldplanes=to_model(row,past)
        out={k:np.tile(a[0],(len(ARMS),1)) for k,a in oldplanes.items()}
        for g in ('fit','direct_fit'):
            out[g]=out['anchored'].copy();out['ref_'+g]=out['ref_anchored'].copy()
        selected=np.full((len(ARMS),2),-1,int); seen=np.zeros_like(selected,dtype=bool)
        ids=np.empty(0,int)
        if row['active'] and len(K):
            j=row_lookup[rid];ids=np.flatnonzero(plan['nom'][j].any(0))
            assert len(ids)<=16
            loc=np.array([lookup[i] for i in ids],int);words=row['words'];nomtotal+=len(ids)
            fields=evidence(plan['p'],ids,plan['boxes'],local['logit'][loc])
            assert digest(row['role'])==row['rh']
            with np.load(row['role']) as z:
                choices=decide(plan,fields,words,plan['us'][j],plan['ur'][j],int(z['side']),int(z['depth']),bool(z['immediate']),m,K)
            for arm,(i,r) in choices.items():
                a=ARMS.index(arm)
                for role,node,c,pre in [(0,i,words[0],''),(1,r,words[1],'ref_')]:
                    if node<0:continue
                    selected[a,role]=node;seen[a,role]=node in ids
                    measured=plan['boxes'][node];anchored=plan['completed'][node,c]
                    out['raw' if role==0 else 'ref'][a]=measured
                    out[pre+'anchored'][a]=anchored
                    fit,direct=anchored,anchored
                    if seen[a,role]:
                        real,erase=local['dims'][lookup[node],:,c]
                        fit,direct=fit_on_node(measured,anchored,real,erase)
                    out[pre+'fit'][a]=fit;out[pre+'direct_fit'][a]=direct
        a=ARMS.index('tree');oldid=old.ARMS.index('shared_cut')
        err=max([err]+[compare_pose(out[k][a],s[oldid]) for k,s in oldplanes.items()])
        assert err<1e-5
        target=WORK/'predictions'/(rid+'.npz');archive_npz(target,**out,seen=seen,selected=selected,nominated=ids)
        result[rid]=digest(target)
    return result,err,nomtotal

def run(src,workers):
    inputs=catalog();plans=read(WORK/'PLANS.json');obs=WORK/'observed'/src/'SEALED.json';received=read(obs)
    assert received['pins']==signatures() and not received['labels_used']
    by=defaultdict(list)
    for r in inputs['rows']:
        if r['source']==src:by[r['frame']].append(r)
    (WORK/'predictions').mkdir(exist_ok=True,mode=0o700)
    tasks=[(f,rows,src,plans['files'].get(f),received['files'].get(f)) for f,rows in by.items()]
    files={};err=0.;n=0;start=perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for outputs,gap,num in pool.map(job,tasks,chunksize=1):
            assert not (files.keys()&outputs.keys());files.update(outputs);err=max(err,gap);n+=num
    save(WORK/(src+'_decisions.json'),dict(files=files,error=err,exposures=n,pins=signatures(),code=code(),
         capture=digest(obs),seconds=perf_counter()-start,labels_used=False))
    print(src,len(files),n,err,flush=True)

def seal():
    rows=catalog()['rows'];files={};source={}
    for s in SOURCES:
        p=WORK/(s+'_decisions.json');record=read(p)
        assert record['pins']==signatures() and record['code']==code() and record['capture']==digest(WORK/'observed'/s/'SEALED.json')
        assert files.keys().isdisjoint(record['files']) and not record['labels_used']
        files.update(record['files']);source[s]=digest(p)
    assert set(files)=={row_id(r) for r in rows}
    for k,h in files.items():assert digest(WORK/'predictions'/(k+'.npz'))==h
    save(WORK/'SEALED.json',dict(files=files,sources=source,protocol=digest(SPEC),code=code(),upstream=base(),
         labels_used=False,ordinary=18620,unparsed_fallback=17326,FINAL_access=False))
    print('sealed',len(files),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=(*SOURCES,'seal'));p.add_argument('--workers',type=int,default=5)
    args=p.parse_args();seal() if args.source=='seal' else run(args.source,args.workers)
