"""Collect two role-conditioned full pools and visible LiDAR supports without evaluation annotations."""
import argparse
from functools import lru_cache
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PARENT=HERE.parent/'3eed_v1_reciprocal_physical_roles_20261120'
sys.path.insert(0,str(PARENT))
from capture_roles import Grounder, SOURCES, observation_list, original_pool
from calibrate import digest, read, save
from predict import archive_npz
from factor import camera
from crossplatform.v1_evidence import visible_pixels
sys.path.insert(0,str(ROOT/'research/round169'))
from panel169 import rewrite

PRIVATE=HERE/'private'
OUT=HERE/'results'
CONFIG=HERE/'PROTOCOL.json'


def signature():
    spec=read(CONFIG)
    assert digest(PARENT/'run_PRIVATE/SEAL.json')==spec['parent_seal']
    assert digest(ROOT/'crossplatform/single_frame_sources.json')==spec['weights']
    return dict(protocol=digest(CONFIG),weights=spec['weights'],parent=spec['parent_seal'],code=digest(Path(__file__)))


def converse(text,info):
    graph,doc=info['graph'],info['document']
    entities={e['role']:i for i,e in enumerate(graph['entities'])}
    edges=set()
    for e in graph['relations']:
        s,r=e['subject'],e['reference']
        if (s=='T')==(r=='T') or e['predicate'] not in ('LEFT','RIGHT'):
            continue
        other=entities.get(r if s=='T' else s)
        if other is not None:
            sign=1 if e['predicate']=='LEFT' else -1
            edges.add((other,sign if s=='T' else -sign))
    if len(edges)!=1:
        return None
    role,sign=next(iter(edges))
    regions={'area','background','bottom','center','edge','foreground','frame','ground','image','lane','middle',
             'picture','region','road','scene','screen','side','sky','top','view'}
    if graph['entities'][role]['head'].lower() in regions:
        return None
    views=rewrite(text,dict(graph=graph,doc=doc),list(edges),role,sign)
    return (sign,views['D']['text']) if views['status']=='supported' else None


def role_mass(frame):
    from reader158 import overlaps
    k=overlaps(frame.boxes)
    return softmax(np.log(np.maximum(frame.probability.astype(float),1.e-300))-np.log(np.sum(k*k,axis=1)))


def annulus(frame,inp):
    # Sort once for inexpensive oriented crops; each deduplicated LiDAR return casts one vote.
    pts=np.unique(frame.sampled_points,axis=0)
    uv,vis=visible_pixels(pts,pts,frame.target,frame.view_index,inp['rgb'],inp['calibration'])
    _,(width,_)=camera(frame.target,frame.view_index,str(inp['calibration']),str(inp['rgb']))
    visible=pts[vis].astype(np.float64)
    columns=np.minimum(width-1,np.maximum(0,np.floor(uv[vis,0]*width).astype(np.int32)))
    return visible,columns,width


def profiles(boxes,visible,columns,width):
    counts=np.zeros((len(boxes),width),np.int32)
    for idx,b in enumerate(boxes):
        co,si=np.cos(b[6]),np.sin(b[6])
        half=b[3:6]*.5+1.e-7
        extent=abs(co)*half[0]+abs(si)*half[1]
        i=np.searchsorted(visible[:,0],b[0]-extent,side='left')
        j=np.searchsorted(visible[:,0],b[0]+extent,side='right')
        delta=visible[i:j]-b[:3]
        inside=(abs(delta[:,2])<=half[2])&(abs(co*delta[:,0]+si*delta[:,1])<=half[0])&(abs(-si*delta[:,0]+co*delta[:,1])<=half[1])
        counts[idx]=np.bincount(columns[i:j][inside],minlength=width)
    return counts


def register_case(model,rec,historical):
    frame=rec['frame']
    cap={}
    main,ev=model._read_observation(frame,capture=cap)
    result=model._decoder(ev)
    old=int(result['choices'][0,0]);jr=int(result['choices'][0,1])
    baseline_s=ev['subject_boxes'][0].double().cpu().numpy()
    baseline_r=ev['reference_boxes'][0].double().cpu().numpy()
    a=original_pool(main)
    np.testing.assert_array_equal(a,historical['pool'])
    np.testing.assert_array_equal(baseline_s,historical['slate'])
    assert old==int(historical['choices'][0])
    valid=bool(ev['has_reference'][0])
    request=converse(frame['text'].strip(),cap) if valid else None
    obs=dict(a=a,mass_a=role_mass(main),native_s=baseline_s,native_r=baseline_r,before=np.array([old,jr]),
             native_ref_valid=valid,active=request is not None)
    if request is None:
        return dict(**obs,sign=0,d=np.empty((0,7)),mass_d=np.empty(0),ha=np.empty((0,0),np.int32),hd=np.empty((0,0),np.int32))
    sign,text=request
    ref,_=model._read_observation(dict(frame,text=text))
    np.testing.assert_array_equal(ref.sampled_points,main.sampled_points)
    d=original_pool(ref)
    pts,col,width=annulus(main,frame)
    return dict(**obs,sign=sign,d=d,mass_d=role_mass(ref),
                ha=profiles(a,pts,col,width),hd=profiles(d,pts,col,width))


def key(rec):
    return f"{rec['panel']}_{rec['case']:03d}_{rec['source']}.npz"


@torch.inference_mode()
def collect(source):
    PRIVATE.mkdir(mode=0o700,exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pin=signature()
    old=read(PARENT/'run_PRIVATE/SEAL.json')
    records=[r for r in observation_list() if r['source']==source]
    model=None
    outputs={}
    torch.set_num_threads(1)
    start=perf_counter()
    for i,r in enumerate(records,1):
        name=key(r); path=PRIVATE/name; receipt=path.with_suffix('.json')
        if receipt.exists():
            p=read(receipt)
            assert p['inputs']==pin and digest(path)==p['hash']
        else:
            if model is None:
                model=Grounder(source)
                model._frontend.parser=lru_cache(maxsize=1024)(model._frontend.parser)
            original=PARENT/'run_PRIVATE'/name
            assert digest(original)==old['files'][name]
            with np.load(original,allow_pickle=False) as history:
                result=register_case(model,r,history)
            archive_npz(path,**result);p=dict(inputs=pin,hash=digest(path));save(receipt,p)
        outputs[name]=p['hash']
        if i%35==0 or i==len(records):
            print(source,i,len(records),round(perf_counter()-start,1),flush=True)
    save(PRIVATE/f'{source}.seal.json',dict(inputs=pin,seconds=perf_counter()-start,files=outputs))


def seal():
    pin=signature();files={}
    for src in SOURCES:
        r=read(PRIVATE/f'{src}.seal.json')
        assert r['inputs']==pin and not r['files'].keys()&files.keys()
        files.update(r['files'])
    assert len(files)==501 and set(files)=={key(r) for r in observation_list()}
    assert all(digest(PRIVATE/f)==h for f,h in files.items())
    save(PRIVATE/'OBSERVED.json',dict(inputs=pin,files=files,n=len(files),labels=False,FINAL_access=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=(*SOURCES,'seal'))
    step=p.parse_args().stage
    seal() if step=='seal' else collect(step)

