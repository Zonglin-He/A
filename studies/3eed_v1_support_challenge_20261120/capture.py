"""Make every four-slot decision from already sealed frozen observations, without task labels."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ORIGINAL=HERE.parent/'3eed_v1_subject_margin_attribution_20261120'/'run_PRIVATE'
WORK=HERE/'private'
OUT=HERE/'results'
PERMS=[(0,1,2,3)]+[p for p in itertools.permutations(range(4)) if all(i!=j for i,j in enumerate(p))]
NAMES=['v1','true']+['perm_'+''.join(map(str,p)) for p in PERMS[1:]]
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    return json.loads(p.read_bytes())
def save(p,obj):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n')
    os.replace(tmp,p)
def bound_components():
    base=ORIGINAL/'SEALED.json'
    assert sha(base)==read(HERE/'PROTOCOL.json')['parent']
    seal=read(base);assert seal['n']==18620 and not seal['target_labels_used'] and not seal['FINAL_access']
    for dr,h in seal['directions'].items():
        d=ORIGINAL/dr;f=d/'SEALED.json';assert sha(f)==h
        yielded=read(f);assert not yielded['labels_used']
        for name,pin in sorted(yielded['files'].items()):
            p=d/name;assert sha(p)==pin
            with np.load(p,allow_pickle=False) as z:
                yield dr,{k:z[k] for k in ('ordinals','slots','prior','unary','rel','final','support','active','choices')}
def support(z):
    x=z['support'][:,1,:4,9:11].astype(float).copy()
    x[:,:,1]*=-1
    assert np.isfinite(x).all()
    return x
def pick(s,q,base):
    current=q[np.arange(len(q)),base]
    credible=(q>current[:,None]).all(2)
    alternate=np.where(credible,s,-np.inf).argmax(1)
    return np.where(credible.any(1),alternate,base)
def main():
    assert len(PERMS)==10
    WORK.mkdir(mode=0o700,exist_ok=True);OUT.mkdir(exist_ok=True)
    storage={}
    for dr,z in bound_components():
        s=z['final'];base=s.argmax(1);np.testing.assert_equal(base,z['choices'][:,0])
        q=support(z)
        picks=np.stack([base]+[pick(s,q[:,p,:],base) for p in PERMS],axis=1).astype('uint8')
        storage.setdefault(dr,[]).append((z['ordinals'],picks))
    pins={};n=0
    for dr,parts in storage.items():
        ix,choices=(np.concatenate(v) for v in zip(*parts));np.testing.assert_equal(ix,np.arange(len(ix)))
        f=WORK/(dr+'.npz');np.savez_compressed(f,ordinals=ix,picks=choices);pins[dr]=sha(f);n+=len(ix)
    assert n==18620
    save(WORK/'SEALED.json',dict(files=pins,total=n,obs=sha(ORIGINAL/'SEALED.json'),protocol=sha(HERE/'PROTOCOL.json'),
        code=sha(Path(__file__)),arms=NAMES,labels_read=False,source_updates=0,target_updates=0,FINAL=False))
    print('sealed',n,len(NAMES))
if __name__=='__main__': main()
