"""Run three frozen observations. Inputs contain the sensor frame and a single expression."""
import argparse
import re
from functools import lru_cache
from hashlib import sha256
from time import perf_counter
import numpy as np
import torch

from paths_witness import ROOT, HERE, WORK, OBS, SPEC, OUT, SOURCES, name, read, save, digest
from collect import Grounder, original_pool, role_mass, annulus, profiles, converse, archive_npz
from projection_witness import projection

PATTERN = re.compile(r'^find the (\w+) immediately to the (left|right) of the (\w+)( closest to the camera| farthest from the camera)? in the camera image\.$',re.I)

def interpret(text,cap):
    matched=PATTERN.fullmatch(text.strip())
    if matched:
        subject,side,ref,qualifier=matched.groups()
        depth=0 if not qualifier else (1 if 'closest' in qualifier.lower() else -1)
        return subject,ref,(-1 if side.lower()=='left' else 1),depth,True
    relation=converse(text,cap)
    if relation is None:return None
    roles={e['role']:e for e in cap['graph']['entities']}
    edges={e['reference'] if e['subject']=='T' else e['subject'] for e in cap['graph']['relations']
           if e['predicate'] in ('LEFT','RIGHT') and 'T' in (e['subject'],e['reference'])}
    if len(edges)!=1:return None
    def phrase(role):
        a,b=roles[role]['span']
        s=text[a:b].strip(' ,.')
        return s if a<b and roles[role]['head'].lower() in s.lower() else ''
    s,r=phrase('T'),phrase(next(iter(edges)))
    if not s or not r:return None
    immediate=bool(re.search(r'\bimmediately (?:to the |on the )?(?:left|right)\b',text,re.I))
    return s,r,-relation[0],0,immediate

def role_text(phrase):
    return re.sub(r'^(?:the|a|an)\s+','',phrase,flags=re.I).strip()

def bank(scene,inp,pts,col,w):
    boxes=original_pool(scene)
    descriptor=projection(boxes,scene.target,scene.view_index,inp['calibration'],inp['rgb'])
    return dict(box=boxes,p=np.asarray(scene.probability,dtype=float),m=role_mass(scene),
                project=descriptor,count=profiles(boxes,pts,col,w).sum(1))

def observe(model,inp,history):
    cap={}
    scene,ev=model._read_observation(inp,capture=cap)
    dec=model._decoder(ev)
    i,j=dec['choices'][0].cpu().tolist()
    output=dict(a=original_pool(scene),ma=role_mass(scene),raw=ev['subject_boxes'][0,i].double().cpu().numpy(),
                ref=ev['reference_boxes'][0,j].double().cpu().numpy() if bool(ev['has_reference'][0]) else np.full(7,np.nan),
                seeds=ev['xyz'][0].double().cpu().numpy())
    parsed=interpret(inp['text'].strip(),cap)
    if parsed is None:return dict(**output,active=False)
    s,r,side,depth,immediate=parsed
    pts,col,width=annulus(scene,inp)
    sig=sha256(scene.sampled_points.tobytes()).hexdigest()
    for prefix,phrase in [('t',s),('r',r)]:
        words='The '+role_text(phrase)+'.'
        key=(str(inp['points']),str(inp['calibration']),int(inp['seed']),str(inp['waymo_view']),words)
        if key in history:
            prev,fields=history[key]; assert prev==sig
        else:
            other,_=model._read_observation(dict(inp,text=words))
            np.testing.assert_array_equal(other.sampled_points,scene.sampled_points)
            fields=bank(other,inp,pts,col,width)
            if len(history)>36:history.clear()
            history[key]=sig,fields
        for k,v in fields.items():output[prefix+'_'+k]=v
    return dict(**output,active=True,side=side,depth=depth,immediate=immediate)

def pins():
    cfg=read(SPEC)
    assert cfg['weights']==digest(ROOT/'crossplatform/single_frame_sources.json')
    assert cfg['current']==digest(ROOT/'crossplatform/CURRENT_METHOD.json')
    return dict(protocol=digest(SPEC),roster=digest(WORK/'roster.json'),
        code={f:digest(HERE/f) for f in ('observe_witness.py','projection_witness.py','paths_witness.py')})

@torch.inference_mode()
def run(source):
    pin=pins();assert pin['roster']==read(OUT/'panel.json')['observations']
    rows=[r for r in read(WORK/'roster.json') if r['source']==source]
    OBS.mkdir(exist_ok=True,mode=0o700)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    model=Grounder(source);model._frontend.parser=lru_cache(maxsize=1500)(model._frontend.parser)
    start=perf_counter(); files={};hist={}
    for i,row in enumerate(rows,1):
        path=OBS/name(row); receipt=path.with_suffix('.json')
        if not receipt.exists():
            result=observe(model,row['frame'],hist)
            archive_npz(path,**result);save(receipt,dict(pin=pin,sha=digest(path)))
        sealed=read(receipt);assert sealed['pin']==pin and digest(path)==sealed['sha']
        files[path.name]=sealed['sha']
        if i%40==0 or i==len(rows):print(source,i,len(rows),round(perf_counter()-start,1),flush=True)
    save(OBS/(source+'.seal.json'),dict(pin=pin,files=files,seconds=perf_counter()-start))

def seal():
    pin=pins(); files={}
    for src in SOURCES:
        record=read(OBS/(src+'.seal.json'))
        assert pin==record['pin'] and not (files.keys() & record['files'].keys())
        files.update(record['files'])
    assert files.keys()=={name(r) for r in read(WORK/'roster.json')}
    assert all(digest(OBS/f)==v for f,v in files.items())
    save(OBS/'SEALED.json',dict(pin=pin,files=files,n=len(files),labels_used=False,FINAL_access=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=(*SOURCES,'seal'))
    stage=parser.parse_args().stage
    seal() if stage=='seal' else run(stage)
