"""Collect only nonfallback role and frozen noun observations required for ordinary Validation."""
import argparse
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from hashlib import sha256
import numpy as np
import torch
from paths_tree import *
from full_roster import FULL,CFG_FULL,full_pins
from scope183 import ordinary,frame,check_input
from collect import Grounder
from observe_witness import observe
from observe_shape import measure,prepare_nouns
from observe_lex import geometry,prompts

def key(f):
    return (str(Path(f['points']).resolve()),int(f['seed']),str(f['waymo_view']),f['target'])

def roles_cached():
    output={};frames={}
    for row,rp,rh,lp,lh,fid,words,*_ in observations():
        f=row['frame'];source=row['source']
        if min(words)<0:continue
        with np.load(rp) as z:
            if not bool(z['active']):continue
        rec=(str(rp),rh,str(lp),lh,fid)
        output[source,key(f),f['text'].strip()]=rec
        frames[source,key(f)]=rec
    return output,frames

def doc_pin():
    p=read(FULL/'ROSTER.json')
    assert p['pin']==full_pins() and p['protocol']==digest(CFG_FULL) and p['parsed']==digest(FULL/'parsed.json')
    return digest(FULL/'ROSTER.json')

@torch.inference_mode()
def capture(source):
    p=doc_pin();parsed=read(FULL/'parsed.json')
    task=ordinary();root=FULL/'observations';root.mkdir(mode=0o700,exist_ok=True)
    folder=root/source;folder.mkdir(mode=0o700,exist_ok=True)
    old,old_frames=roles_cached();old_words=read(parent.OBS/'SEALED.json')['hashes']
    records={};frame_records={}
    history={};checked=[];start=perf_counter();n=0;old_n=0;fresh_frames=0
    model=Grounder(source)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    front=model._frontend; front.parser=lru_cache(maxsize=128)(front.parser)
    prepared=prepare_nouns(front.base);multi=prompts(front.base)
    for target,table in task.items():
        if source==target:continue
        for ix,row in enumerate(table):
            spec=parsed[target][ix]
            if spec is None:continue
            check_input(row)
            f=frame(row);kf=key(f);rec=old.get((source,kf,f['text'].strip()))
            h=old_frames.get((source,kf));tag=target+'_'+str(ix)
            if h:
                _,_,lp,lh,fid=h
                wp=parent.OBS/(fid+'.npz');wh=old_words[fid]
            else:
                fid=sha256(repr(('ordinary-physical-cuts/1',source,kf)).encode()).hexdigest()
                lp=folder/(fid+'_lex.npz');wp=folder/(fid+'_word.npz')
                receipt=folder/(fid+'.json')
                if receipt.exists():
                    existing=read(receipt);assert existing['pin']==p and existing['code']==digest(HERE/'full_observe.py')
                    lh,wh=existing['lex'],existing['word']
                else:
                    l=geometry(f,front.base,multi,not checked);w=measure(front.base,f,prepared,not checked)
                    np.testing.assert_array_equal(l['xyz'],w['xyz'])
                    if not checked:checked.append([float(max(l['exact_native_check'])),float(max(w['native_error']))])
                    archive_npz(lp,**l);archive_npz(wp,**w);lh,wh=digest(lp),digest(wp)
                    save(receipt,dict(pin=p,code=digest(HERE/'full_observe.py'),lex=lh,word=wh))
                    fresh_frames+=1
            if fid not in frame_records:
                assert digest(lp)==lh and digest(wp)==wh
                frame_records[fid]=dict(lex=str(lp),word=str(wp),lex_sha=lh,word_sha=wh)
            if rec:
                rp,rh,*_=rec;old_n+=1
            else:
                rp=folder/(tag+'_role.npz');receipt=folder/(tag+'_role.json')
                if receipt.exists():
                    existing=read(receipt);assert existing['pin']==p and existing['code']==digest(HERE/'full_observe.py')
                    rh=existing['sha']
                else:
                    z=observe(model,f,history)
                    assert z['active'] and spec[2:]==[int(z['side']),int(z['depth']),bool(z['immediate'])]
                    with np.load(wp) as w:np.testing.assert_array_equal(z['seeds'],w['xyz'])
                    archive_npz(rp,**z);rh=digest(rp)
                    save(receipt,dict(pin=p,code=digest(HERE/'full_observe.py'),sha=rh))
            assert digest(rp)==rh
            records[tag]=dict(role=str(rp),sha=rh,frame=fid)
            n+=1
            if n%100==0:print(source,'records',n,'fresh scenes',fresh_frames,round(perf_counter()-start,1),flush=True)
    save(root/(source+'.json'),dict(pin=p,protocol=digest(CFG_FULL),code=digest(HERE/'full_observe.py'),
        records=records,frames=frame_records,archived_rows=old_n,checks=checked,seconds=perf_counter()-start,
        labels_used=False,FINAL_access=False))
    print('done',source,n,fresh_frames,old_n,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source',choices=SOURCES);capture(p.parse_args().source)
