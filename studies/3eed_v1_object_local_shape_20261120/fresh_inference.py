"""Run each new single-word trial independently before the scoring inventory is opened."""
import argparse
from functools import lru_cache

import numpy as np
import torch

from local_paths import *
from method_local import ARMS, GEOMS, decide, evidence, fit_on_node
from local_reader import LocalReader, observe_objects
from nominate import cover
from role_cut import structure, shared
from physical_nodes import assemble, overlap_prior
from physical_evidence import Returns
from scene_partition import terrain, build
from scene_faces import bottom, only_ground, split_returns
from observe_witness import observe
from observe_shape import measure, prepare_nouns
from observe_lex import geometry, prompts, nouns
from full_observe import key
from scope183 import VIEWS
from collect import Grounder

FRESH = WORK/'fresh'
def pins():
    return dict(protocol=digest(HERE/'FRESH_PROTOCOL.json'), input=digest(FRESH/'input.json'),
                main=digest(SPEC), implementations={name:digest(HERE/name) for name in
                ('fresh_inference.py','local_reader.py','nominate.py','method_local.py','ROLE_CONTROL.json')})

def role_read(data, native, words, reader, cache):
    mass, kernel = structure(data['parent'], data['stop'], data['size'])
    if not len(kernel):
        return {}, {}, np.empty(0,int)
    points=Returns(data['xyz'])
    us=overlap_prior(data,native,'t',points)
    ur=overlap_prior(data,native,'r',points)
    side,depth,near=int(native['side']),int(native['depth']),bool(native['immediate'])
    visible=np.isfinite(data['view'][:,0])
    joint, ref=shared(data['parent'],kernel==0,data['stop'],data['p'],words,data['view'],side,depth,near,'physical')
    pair=joint*(us[:,None]*ur)
    s,r=pair.sum(1),pair.sum(0)
    if s.sum()==0:s=us*mass*data['p'][:,words[0]]*visible
    if r.sum()==0:r=ur*ref
    ids=np.array(sorted(set(cover(s,kernel,visible))|set(cover(r,kernel,visible))),int)
    missing=[i for i in ids if i not in cache]
    if missing:
        crops=[(data['xyz'][(data['assign']==i).any(0)],data['boxes'][i,6]) for i in missing]
        assert all(len(x)==data['size'][i] for i,(x,_) in zip(missing,crops))
        z=observe_objects(reader,crops)
        cache.update({i:(z['logit'][j],z['dims'][j]) for j,i in enumerate(missing)})
    logits=np.array([cache[i][0] for i in ids]).reshape(-1,2,7)
    fields=evidence(data['p'],ids,data['boxes'],logits)
    readings=decide(data,fields,words,us,ur,side,depth,near,mass,kernel)
    return readings,{i:cache[i][1] for i in ids},ids

@torch.inference_mode()
def run(src):
    cert=read(OUT/'fresh_panel.json');signature=pins()
    assert cert['roster']==signature['input'] and cert['protocol']==signature['protocol']
    rows=[x for x in read(FRESH/'input.json') if x['source']==src]
    dest=FRESH/src;dest.mkdir(mode=0o700,exist_ok=True)
    files={}
    if rows:
        torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
        engine=Grounder(src);front=engine._frontend;model=front.base
        front.parser=lru_cache(maxsize=64)(front.parser)
        multi=prompts(model);lexical=prepare_nouns(model);reader=LocalReader(model)
        cache={};history={}
    for row in rows:
        f=row['frame'];fid=key(f)
        if fid not in cache:
            obs=geometry(f,model,multi,not cache);word=measure(model,f,lexical,not cache)
            data=assemble(build(obs['sample'],obs['logits'],obs['xyz'],f),word)
            cache[fid]=(data,{},np.unique(obs['sample'].astype(float),axis=0))
        data,local,x=cache[fid]
        z=observe(engine,f,history);words=list(nouns(f,front))[:2]
        assert z['active'] and min(words)>=0
        views=VIEWS.index(f['waymo_view']);anchor=z['raw']
        ground=only_ground(anchor,bottom(anchor,x,terrain(x),split_returns(x)))
        ans={'raw':np.tile(anchor,(len(ARMS),1)),'ref':np.tile(z['ref'],(len(ARMS),1))}
        for g in GEOMS[1:]:
            ans[g]=np.tile(ground,(len(ARMS),1));ans['ref_'+g]=ans['ref'].copy()
        selected=np.full((len(ARMS),2),-1,int)
        choices,dims,ids=role_read(data,z,words,reader,local)
        for arm,(s,r) in choices.items():
            k=ARMS.index(arm)
            for role,(node,cl) in enumerate(zip((s,r),words)):
                if node<0:continue
                selected[k,role]=node
                box=data['boxes'][node];basebox=data['completed'][node,cl]
                fit,direct=fit_on_node(box,basebox,*dims[node][:,cl]) if node in dims else (basebox,basebox)
                names=GEOMS if role==0 else ('ref','ref_anchored','ref_fit','ref_direct_fit')
                for name,value in zip(names,(box,basebox,fit,direct)):ans[name][k]=value
        p=dest/(row['uid']+'.npz')
        archive_npz(p,**ans,selected=selected,nominated=ids,seeds=z['seeds'],view=np.array(views),native_a=z['a'])
        files[row['uid']]=digest(p)
    save(dest/'SEALED.json',dict(pins=signature,files=files,labels_read=False,updates=0))
    print('fresh',src,len(files),flush=True)
def seal():
    p=pins();rows=read(FRESH/'input.json');n=0;hashes={}
    for s in SOURCES:
        f=FRESH/s/'SEALED.json';z=read(f);assert z['pins']==p and not z['labels_read']
        assert {r['uid'] for r in rows if r['source']==s}==set(z['files'])
        for k,h in z['files'].items():assert digest(FRESH/s/(k+'.npz'))==h
        hashes[s]=digest(f);n+=len(z['files'])
    assert n==read(OUT/'fresh_panel.json')['direction_records']
    save(FRESH/'SEALED.json',dict(n=n,sources=hashes,labels_read=False,pins=p))
    print('sealed new records',n)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=[*SOURCES,'seal']);a=p.parse_args()
    seal() if a.stage=='seal' else run(a.stage)
