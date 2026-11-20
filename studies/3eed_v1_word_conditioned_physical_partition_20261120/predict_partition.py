"""Shared inputs, fixed physical controls and pre-evaluation prediction seals."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import numpy as np

from partition_paths import ARMS, GEOMS, WORK, archive, archive_npz, catalog, digest, read, row_id, save, signed
from physical_partition import kernel, partition
from physical_evidence import Returns
from physical_nodes import overlap_prior
from predict_delivery import to_model
from role_cut import structure, shared, medoid, build_library


def controls(data):
    mark=data['mark'][:,:6];p=data['p'];stop=data['stop']
    same=np.zeros_like(mark)
    for c,n in enumerate(mark.sum(0)):
        order=np.argsort(-stop*p[:,c],kind='stable')
        same[order[:n],c]=True
    boxes=data['boxes']
    order=np.lexsort((boxes[:,2],boxes[:,1],boxes[:,0]))
    wrong=mark.copy()
    wrong[order]=mark[np.roll(order,len(order)//2)]
    np.testing.assert_equal(same.sum(0),mark.sum(0))
    np.testing.assert_equal(wrong.sum(0),mark.sum(0))
    masks=dict(union=np.ones_like(mark),blind=np.repeat(data['mark'][:,6:],6,axis=1),
               same_count=same,wrong_marks=wrong,word_cut=mark)
    fields={}
    _,K=structure(data['parent'],stop,data['size'])
    for name,mask in masks.items():
        emission=stop[:,None]*p*mask
        total=emission.sum(1)
        conditional=np.divide(emission,total[:,None],out=np.zeros_like(p),where=total[:,None]>0)
        survive=np.ones_like(total)
        for v in range(len(total)-1,-1,-1):
            parent=data['parent'][v]
            if parent>=0:survive[v]=survive[parent]*(1.-total[parent])
        fields[name]=(total,conditional,survive*total)
    return K,fields


def outputs(base):
    result={}
    for role,raw,completed in [('', 'raw','anchored'),('ref_','ref','ref_anchored')]:
        for g in GEOMS:
            source=base[raw if g=='measured' else completed]
            result[role+g]=np.tile(source[0],(len(ARMS),1))
            result[role+g][ARMS.index('previous_tree')]=source[1]
    result['selected']=np.full((len(ARMS),2),-1,np.int32)
    return result


def decision(data,ctl,native,words,result,points):
    K,fields=ctl
    if not len(K):return result
    valid=K==0
    us=overlap_prior(data,native,'t',points);ur=overlap_prior(data,native,'r',points)
    visible=np.isfinite(data['view'][:,0]);s,r=words
    side,depth,nearest=int(native['side']),int(native['depth']),bool(native['immediate'])
    def compute(stop,p,mode):
        return shared(data['parent'],valid,stop,p,words,data['view'],side,depth,nearest,mode)
    def record(arm,choice):
        a=ARMS.index(arm)
        for role,(v,c,pre) in enumerate(zip(choice,words,('','ref_'))):
            if v<0:continue
            result['selected'][a,role]=v
            result[pre+'measured'][a]=data['boxes'][v]
            for g in GEOMS[1:]:result[pre+g][a]=data[g][v,c]
    for arm,(stop,p,occ) in fields.items():
        joint,ref=compute(stop,p,'physical')
        record(arm,medoid(joint,K,us,ur,valid)[:2])
        if arm=='word_cut':
            def pick(score,emitted):
                allowed=visible & (emitted>0)
                return int(np.argmax(np.where(allowed,score,-1))) if np.any(allowed & (score>0)) else -1
            sm=us*occ*p[:,s];rm=ur*ref
            record('independent',(pick(K@(sm*visible),sm),pick(K@rm,rm)))
            other,_=compute(stop,p,'wrong_reference_binding')
            record('wrong_reference',medoid(other,K,us,ur,valid)[:2])
    return result


def job(task):
    fid,info,rows,config=task
    receipt=WORK/'frame_seals'/(fid+'.json')
    pins=dict(frame=[info.get('lh'),info.get('wh')] if info else None,
              rows={row_id(r):[r['ph'],r['rh']] for r in rows},code=config)
    if receipt.exists():
        saved=read(receipt)
        if saved['pins']==pins and all(digest(WORK/'decisions'/(rid+'.npz'))==h for rid,h in saved['files'].items()):
            return saved['files'],saved['nodes']
    data=None;files={}
    if any(r['active'] for r in rows):
        assert info and digest(info['lex'])==info['lh'] and digest(info['word'])==info['wh']
        with np.load(info['lex']) as lex,np.load(info['word']) as word:data=partition(lex,word,info['f'])
        ctl=controls(data);points=Returns(data['xyz'])
        assert np.array_equal(points.xyz,data['xyz'])
        keep=('boxes','local','scene_pca','envelope','mark','parent','size','view')
        plan=WORK/'plans'/(fid+'.npz')
        archive_npz(plan,**{k:data[k] for k in keep})
    for row in rows:
        rid=row_id(row)
        assert digest(row['prior'])==row['ph']
        with np.load(row['prior']) as past:
            old=to_model(row,past)
            result=outputs({k:v[[0,archive.old.ARMS.index('shared_cut')]] for k,v in old.items()})
        if row['active']:
            assert digest(row['role'])==row['rh']
            with np.load(row['role']) as native:
                decision(data,ctl,native,row['words'],result,points)
        path=WORK/'decisions'/(rid+'.npz');archive_npz(path,**result)
        files[rid]=digest(path)
    nodes=0 if data is None else len(data['parent'])
    save(receipt,dict(pins=pins,files=files,nodes=nodes,plan=digest(plan) if data is not None else None))
    return files,nodes


def run(workers,scope):
    cfg=signed();kernel();build_library();roster=catalog();groups=defaultdict(list)
    for d in ('frame_seals','plans','decisions'):(WORK/d).mkdir(exist_ok=True,mode=0o700)
    for row in roster['rows']:groups[row['frame']].append(row)
    def category(rows):
        return min({'noun':0,'panel':1,'ordinary':2}[r['scope']] for r in rows)
    jobs=[(k,roster['frames'].get(k),rows,cfg) for k,rows in groups.items()]
    jobs.sort(key=lambda j:(category(j[2]),j[0]))
    if scope=='noun':jobs=[j for j in jobs if any(r['scope']=='noun' for r in j[2])]
    start=perf_counter();files={};nodes=[];seals={}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for j,(p,n) in enumerate(pool.map(job,jobs,chunksize=1)):
            assert files.keys().isdisjoint(p);files.update(p);nodes.append(n)
            fid=jobs[j][0];seals[fid]=digest(WORK/'frame_seals'/(fid+'.json'))
            if (j+1)%80==0:print('frames',j+1,'/',len(jobs),'seconds',round(perf_counter()-start,1),flush=True)
    assert set(files)=={row_id(r) for _,_,rows,_ in jobs for r in rows}
    save(WORK/(('SEALED' if scope=='all' else 'NOUN_SEALED')+'.json'),dict(code=cfg,files=files,frame_seals=seals,
        frames=len(jobs),max_nodes=max(nodes),seconds=perf_counter()-start,labels_used=False,FINAL_access=False))
    print('sealed',scope,len(jobs),len(files),max(nodes),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=6);p.add_argument('--scope',choices=['noun','all'],default='all')
    a=p.parse_args();run(a.workers,a.scope)
