"""Independent single-utterance capture for the single-word landmark intervention."""
from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import argparse
import numpy as np
import torch
from panel_nouns import EXTRA,PIN
from paths_tree import *
from scene_faces import bottom,split_returns,only_ground
from scene_partition import terrain
from common_lex import sample
from collect import Grounder
from observe_witness import observe
from observe_shape import measure,prepare_nouns
from observe_lex import geometry,prompts,nouns
from full_decide import case
from full_observe import key
from scope183 import VIEWS
from score_roles import sensor_frame_box

def inputs():
    cert=read(OUT/'noun_panel.json')
    assert cert['protocol']==digest(PIN) and cert['code']==digest(HERE/'panel_nouns.py')
    assert cert['observations']==digest(EXTRA/'roster.json') and read(PIN)['parent']==digest(PRED/'SEALED.json')
    return read(EXTRA/'roster.json')

@torch.inference_mode()
def capture(source):
    selected=[r for r in inputs() if r['source']==source]
    folder=EXTRA/source;folder.mkdir(mode=0o700,exist_ok=True)
    if not selected:
        save(folder/'SEALED.json',dict(n=0,code=digest(HERE/'predict_nouns.py'),files={},protocol=digest(PIN),label_free=True))
        return
    engine=Grounder(source);front=engine._frontend;front.parser=lru_cache(maxsize=64)(front.parser)
    model=front.base;torch.set_num_threads(1);torch.backends.cudnn.benchmark=False
    multi=prompts(model);lexical=prepare_nouns(model)
    cache={};history={};files={};syn=0.
    for count,row in enumerate(selected,1):
        f=row['frame'];target=f['target'];kf=key(f)
        if kf not in cache:
            fid=sha256(repr(('noun-only-physical/1',source,kf)).encode()).hexdigest()
            l=geometry(f,model,multi,not cache);w=measure(model,f,lexical,not cache)
            lp=folder/(fid+'_lex.npz');wp=folder/(fid+'_word.npz')
            archive_npz(lp,**l);archive_npz(wp,**w)
            cache[kf]=(fid,dict(lex=str(lp),word=str(wp),lex_sha=digest(lp),word_sha=digest(wp)))
        fid,wordmeta=cache[kf];z=observe(engine,f,history)
        words=list(nouns(f,front));assert z['active'] and min(words)>=0
        with np.load(wordmeta['word']) as w:np.testing.assert_equal(z['seeds'],w['xyz'])
        rp=folder/(name(row)[:-4]+'_native.npz');archive_npz(rp,**z)
        x=np.unique(sample(f).astype(float),axis=0);plane=terrain(x);sp=split_returns(x)
        anchor=z['raw'];geo=only_ground(anchor,bottom(anchor,x,plane,sp))
        view=VIEWS.index(f['waymo_view'])
        anchors=np.array([sensor_frame_box(b,target,view) for b in (anchor,geo)])
        _,info,err,_=case((source,target,count,f,words,dict(role=str(rp),sha=digest(rp),frame=fid),wordmeta,anchors,None))
        syn=max(syn,err)
        r=folder/name(row);archive_npz(r,**info)
        files[name(row)]=dict(pred=digest(r),native=digest(rp),lex=wordmeta['lex_sha'],word=wordmeta['word_sha'],fid=fid)
        if count%16==0:print('single-word',source,count,len(selected),flush=True)
    save(folder/'SEALED.json',dict(n=len(files),files=files,code=digest(HERE/'predict_nouns.py'),
         run_code=digest(HERE/'full_decide.py'),protocol=digest(PIN),max_v1_error=syn,label_free=True))

def seal():
    data={}
    for s in SOURCES:
        check=read(EXTRA/s/'SEALED.json');assert check['label_free'] and check['protocol']==digest(PIN)
        assert check['code']==digest(HERE/'predict_nouns.py');data[s]=digest(EXTRA/s/'SEALED.json')
    n=sum(read(EXTRA/s/'SEALED.json')['n'] for s in SOURCES)
    assert n==len(inputs())
    save(EXTRA/'SEALED.json',dict(n=n,sources=data,protocol=digest(PIN),small=digest(PRED/'SEALED.json'),
         implementation=digest(HERE/'full_decide.py'),labels_used=False,FINAL_access=False))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=(*SOURCES,'seal'))
    args=p.parse_args()
    seal() if args.stage=='seal' else capture(args.stage)
