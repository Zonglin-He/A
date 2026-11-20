"""Generate observation-only applicability from the production dependency parser for every caption."""
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import argparse
import sys
import numpy as np
from paths_tree import *
from observe_witness import interpret
from roles import nominal
from scope183 import ordinary,frame,INPUT
from terrain_followup import DST
CFG_FULL=OUT/'full_protocol.json'
FULL=WORK/'ordinary'
def full_pins():
    cfg=read(CFG_FULL)
    pins=dict(previous_seal=digest(PRED/'SEALED.json'),ordinary_roster=digest(INPUT/'SEALED.json'),terrain_seal=digest(DST/'SEALED.json'))
    assert all(cfg[k]==v for k,v in pins.items())
    return pins

@lru_cache(maxsize=1)
def parser():
    import torch
    sys.path.insert(0,str(ROOT/'research/round97'))
    import parse97
    torch.set_num_threads(1)
    assets=ROOT/'crossplatform/round20/assets'
    kw=read(assets/'STANZA_ASSET_RECEIPT.json')['kwargs']
    opts={k:str(assets/v) if k.endswith('_path') else v for k,v in kw.items()}
    return parse97.stanza.Pipeline('en',dir=str(assets/'stanza'),processors='tokenize,mwt,pos,lemma,depparse',
        package=None,use_gpu=False,download_method=None,verbose=False,**opts)

def parsed(text):
    nlp=parser()
    from parse97 import dependency_document,parse85,parse_subject,repair_inventory87
    from observe_witness import PATTERN
    if PATTERN.fullmatch(text.strip()):
        r=interpret(text,{})
    else:
        doc=dependency_document(nlp(text))
        graph=repair_inventory87(text,doc,parse85(text,doc,parse_subject(text,doc)))
        r=interpret(text,dict(graph=graph,document=doc))
    if r is None:return None
    words=[nominal(p) for p in r[:2]]
    words=[{'othervehicle':'other vehicle'}.get(k,k) for k in words]
    if any(k not in CLASSES[:-1] for k in words):return None
    return [CLASSES.index(k) for k in words]+[int(r[2]),int(r[3]),bool(r[4])]

def make(workers):
    full_pins();FULL.mkdir(mode=0o700,exist_ok=True)
    roster=ordinary()
    texts=sorted({r['caption'].strip() for v in roster.values() for r in v})
    with ProcessPoolExecutor(max_workers=workers) as ex:
        alltext={}
        for i,(t,parsed_) in enumerate(zip(texts,ex.map(parsed,texts,chunksize=24)),1):
            alltext[t]=parsed_
            if i%1500==0:print('parsed',i,len(texts),flush=True)
    by={target:[alltext[r['caption'].strip()] for r in rows] for target,rows in roster.items()}
    save(FULL/'parsed.json',by)
    overlap=0
    for row,path,h,_,_,_,words,*_ in observations():
        f=row['frame']
        if f['text'].strip() not in alltext:continue
        assert digest(path)==h
        with np.load(path) as z:
            if bool(z['active']) and min(words)>=0:
                obs=list(words)+[int(z['side']),int(z['depth']),bool(z['immediate'])]
            else:obs=None
            assert obs==alltext[f['text'].strip()]
        overlap+=1
    save(FULL/'ROSTER.json',dict(pin=full_pins(),protocol=digest(CFG_FULL),code=digest(HERE/'full_roster.py'),
         parsed=digest(FULL/'parsed.json'),n=sum(map(len,by.values())),historical_parse_checks=overlap,
         active={s:sum(k is not None for k in v) for s,v in by.items()},labels_used=False))
    print('eligible',read(FULL/'ROSTER.json')['active'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=5);make(p.parse_args().workers)
