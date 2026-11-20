"""Observation-only replay. All rules share physical candidates, priors, cuts, and random draws."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import numpy as np
from paths_shape import *
from scene_partition import build
from scene_roles import world, role, infer, marginal_roles, anchor_returns
from scene_faces import bottom, only_ground, split_returns
from physical_evidence import Returns
from ordered_choice import prepare
from shape_evidence import prepare as shape_fields

CODE=('paths_shape.py','shape_evidence.py','predict_shape.py')
FIELDS=WORK/'fields'


def observations():
    seal=read(OBS/'SEALED.json')
    rows=list(previous.inputs())
    for row in read(WORK/'roster.json'):
        n=name(row);v=seal['files'][n];fid=v['fid']
        rows.append((row,WORK/'role'/n,v['role'],WORK/'lexical'/(fid+'.npz'),v['lex'],fid,v['words'],None,None))
    return rows


def prepare_frame(task):
    fid,path,sha,f,old,required,wordsha=task
    dest=FIELDS/(fid+'.npz')
    if old:
        segfile=previous.SEG/(fid+'.npz')
        assert digest(segfile)==old
        with np.load(segfile) as src:
            seg={k:src[k] for k in src}
    else:
        assert digest(path)==sha
        with np.load(path) as h:
            seg=build(h['sample'],h['logits'],h['xyz'],f)
        archive_npz(WORK/'segments'/(fid+'.npz'),**seg)
    answer={}
    if required:
        wordfile=OBS/(fid+'.npz')
        assert digest(wordfile)==wordsha
        with np.load(wordfile) as w:
            answer=shape_fields(seg,w)
    archive_npz(dest,**answer)
    return fid,(digest(dest),old or digest(WORK/'segments'/(fid+'.npz')))


def fields():
    seal=read(OBS/'SEALED.json')
    pins={}
    for part in ('old','new'):
        pins.update(read(previous.SEG/(part+'.json'))['files'])
    tasks={}
    for row,_,_,lp,lh,fid,*_ in observations():
        required=fid in seal['hashes']
        tasks[fid]=(fid,lp,lh,row['frame'],pins.get(fid),required,seal['hashes'].get(fid))
    FIELDS.mkdir(exist_ok=True,mode=0o700)
    (WORK/'segments').mkdir(exist_ok=True,mode=0o700)
    with ProcessPoolExecutor(max_workers=7) as ex:
        results=dict(ex.map(prepare_frame,tasks.values(),chunksize=2))
    save(FIELDS/'SEALED.json',dict(obs=digest(OBS/'SEALED.json'),code=digest(HERE/'shape_evidence.py'),files=results))


def extreme(tables,side,nt):
    scores=np.zeros(nt)
    for t,_ in tables:
        if not len(t.index):continue
        # Extreme on the stated side, among eligible same-noun components in each lexical draw.
        pos=-side*t.view[:,1]
        limits=np.where(t.observed,pos,np.inf).min(1,keepdims=True)
        w=t.prior*(t.observed & (pos==limits))
        denom=w.sum(1,keepdims=True)
        w=np.divide(w,denom,out=np.zeros_like(w),where=denom>0)
        scores+=t.assign@w.mean(0)/len(tables)
    return scores


def run_case(z,seg,extra,words,seed):
    n=len(ARMS)
    raw=np.tile(z['raw'],(n,1));refs=np.tile(z['ref'],(n,1));active=np.zeros(n,bool)
    x=seg['xyz'].astype(float);split=split_returns(x);T=np.empty((0,7));R=T
    ranks=np.zeros((n,0))
    def result():
        unique,inverse=np.unique(raw,axis=0,return_inverse=True)
        after=np.asarray([only_ground(b,bottom(b,x,seg['plane'],split)) for b in unique])[inverse]
        assert np.isfinite(after).all() and (after[:,3:6]>0).all()
        return dict(raw=raw,ref=refs,final=after,applied=active,candidates=T,ref_candidates=R,
                    scores=ranks,coverage=extra.get('coverage',np.zeros((8,4))))
    if not z['active'] or min(words)<0:return result()
    T,_,mt,_=prepare(z,'t');R,_,mr,_=prepare(z,'r')
    ranks=np.zeros((n,len(T)))
    if not len(T) or not len(R):return result()
    points=Returns(x)
    rt,rr=anchor_returns(T,points),anchor_returns(R,points)
    side,depth,nearest=int(z['side']),int(z['depth']),bool(z['immediate'])
    for kind,arms in [('original',['frozen_words']),('global',['scene_template']),('wrong',['wrong_noun']),
                      ('local',['subject_only','local_shape','wrong_reference','image_center','image_extreme'])]:
        w=dict(seg)
        if kind!='original':
            for k in range(8):w['p'+str(k)]=extra[kind+str(k)]
        tables=[]
        for k in range(8):
            scene=world(w,k,seed)
            tables.append((role(rt,mt,words[0],scene),role(rr,mr,words[1],scene)))
        for arm in arms:
            i=ARMS.index(arm)
            if arm in ('subject_only','image_extreme'):
                ts,rs=marginal_roles(tables,depth,len(T),len(R))
                if arm=='image_extreme':ts=extreme(tables,side,len(T))
                ranks[i]=ts
                if ts.sum()>0:
                    raw[i]=T[ts.argmax()];active[i]=True
                if rs.sum()>0:refs[i]=R[rs.argmax()]
                continue
            control=arm if arm in ('wrong_reference','image_center') else 'aligned'
            J,_=infer(tables,seg['labels'],side,depth,nearest,control,len(T),len(R))
            ranks[i]=J.sum(1)
            if J.sum()>0:
                b=int(ranks[i].argmax())
                raw[i]=T[b];refs[i]=R[J[b].argmax()];active[i]=True
    return result()


def predict_one(task):
    row,path,sha,fid,words,sfile,sh,fh=task
    assert digest(path)==sha and digest(sfile)==sh and digest(FIELDS/(fid+'.npz'))==fh
    with np.load(path) as z,np.load(sfile) as data,np.load(FIELDS/(fid+'.npz')) as f:
        seg={k:data[k] for k in data}
        output=run_case(z,seg,dict(f),words,int(fid[:8],16))
    dst=PRED/name(row);archive_npz(dst,**output)
    return dst.name,digest(dst)


def predict():
    rows=observations()
    fields=read(FIELDS/'SEALED.json')
    assert fields['obs']==digest(OBS/'SEALED.json') and fields['code']==digest(HERE/'shape_evidence.py')
    tasks=[]
    for row,path,sha,_,_,fid,words,*_ in rows:
        s=previous.SEG/(fid+'.npz')
        if not s.exists():s=WORK/'segments'/(fid+'.npz')
        fh,sh=fields['files'][fid]
        tasks.append((row,path,sha,fid,words,s,sh,fh))
    PRED.mkdir(mode=0o700,exist_ok=True)
    files={}
    with ProcessPoolExecutor(max_workers=7) as ex:
        for i,(n,h) in enumerate(ex.map(predict_one,tasks,chunksize=2),1):
            files[n]=h
            if i%200==0:print('predictions',i,len(rows),flush=True)
    assert set(files)=={name(r[0]) for r in rows}
    save(PRED/'SEALED.json',dict(n=len(rows),protocol=digest(SPEC),fields=digest(FIELDS/'SEALED.json'),
        observations=digest(OBS/'SEALED.json'),code={n:digest(HERE/n) for n in CODE},
        labels_used=False,target_updates=0,source_updates=0,FINAL_access=False,files=files))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fields','predict'])
    fields() if p.parse_args().stage=='fields' else predict()
