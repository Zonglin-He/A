"""Test the exact binary-predicate reduction without looking at ground-truth outcomes."""
from collections import Counter,defaultdict
import re
import numpy as np
from paths_tree import *
from role_cut import shared,structure,layout
from full_roster import FULL,full_pins
from scope183 import ordinary
from score_roles import csv_file

def factor(parent,stop,n):
    anc=np.eye(n,dtype=bool)
    for i in reversed(range(n)):
        if parent[i]>=0:anc[i]|=anc[parent[i]]
    np.fill_diagonal(anc,False)
    gain=np.zeros((n,n))
    for v in range(n):
        d=np.flatnonzero(anc[:,v])
        if len(d):gain[np.ix_(d,d)]+=-np.log1p(-stop[v])
    return gain

def run():
    seal=read(OBJECTS/'SEALED.json');assert seal['input']==base_pins() and not seal['labels_used']
    roster=read(FULL/'ROSTER.json')
    assert roster['pin']==full_pins() and roster['parsed']==digest(FULL/'parsed.json')
    cfg=read(FULL/'parsed.json');rows=ordinary()
    counts=[]
    for tgt,rr in cfg.items():
        sel=[(r,original['caption']) for r,original in zip(rr,rows[tgt]) if r is not None]
        counts.append(dict(target=tgt,applicable_descriptions=len(sel),active_directions=2*len(sel),
                    immediate=sum(bool(r[4]) for r,_ in sel),depth=sum(bool(r[3]) for r,_ in sel),
                    mentions_nearest_or_farthest=sum(bool(re.search(r'\b(closest|farthest|nearest|immediately)\b',s,re.I))
                                                     for _,s in sel)))
    csv_file(OUT/'ordinary_relation_forms.csv',counts)
    seen=set();pairs=0;common=0;w=0.;ungained=0.;maxerr=0.;nonzero=0
    for row,role,h,_,_,fid,words,*_ in observations():
        if fid in seen or fid not in seal['files'] or min(words)<0:continue
        assert digest(role)==h
        with np.load(role) as z:
            if not z['active']:continue
            side=int(z['side'])
        seen.add(fid);file=OBJECTS/(fid+'.npz');assert digest(file)==seal['files'][fid]
        with np.load(file) as d:
            p=d['p'];parent=d['parent'];stop=d['stop'];views=d['view']
            assert np.all(stop<1)
            m,K=structure(parent,stop,d['size']);ok=K==0
            gain=factor(parent,stop,len(m))
            _,_,vis,gap=layout(views,side,'physical')
            mask=ok & (gap>0) & vis[:,None] & vis[None,:]
            base=(m*p[:,words[0]])[:,None]*(m*p[:,words[1]])
            expected=base*np.exp(gain)*mask
            actual,_=shared(parent,ok,stop,p,words,views,side,0,False,'physical')
            np.testing.assert_allclose(expected,actual,atol=1e-12,rtol=1e-10)
            maxerr=max(maxerr,float(np.max(abs(expected-actual),initial=0)))
            pairs+=int(mask.sum());common+=int((mask&(gain>0)).sum())
            w+=float(expected.sum());ungained+=float(expected[gain==0].sum())
            nonzero+=int(np.any(mask&(gain>0)))
    save(OUT/'binary_relation_factorization.json',dict(frame_sources=len(seen),feasible_pairs=pairs,
         pairs_with_common_positive_stop_ancestor=common,scenes_with_any_common_ancestor=nonzero,
         unnormalized_witness_mass_disjoint_roots_fraction=ungained/w,max_kernel_error=maxerr,
         interpretation='One frozen eligible word pair per frame-source, label free. Subject/ref proposal priors are omitted in this algebra check.'))
if __name__=='__main__':run()
