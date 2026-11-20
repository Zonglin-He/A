"""Post-seal exact dominance, paired accuracy and within-recording resampling."""
import csv
import gzip
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from capture import HERE, ORIGINAL, WORK, OUT, PERMS, NAMES, bound_components, read, save, sha, support
sys.path.insert(0,str(HERE.parent/'3eed_v2_ordinary_val_subject_ranking_20261119'))
from scope183 import ordinary

def has_undominated(x,good):
    # Order is [record, possible dominator, possible passing slot, component].
    worse_or_equal=(x[:,:,None,:]>=x[:,None,:,:]).all(-1)
    dominated=(worse_or_equal & ~good[:,:,None]).any(1)
    return (good & ~dominated).any(1)
def independent_check(x,good):
    for g in np.flatnonzero(good):
        if all(any(x[g,i]>x[b,i] for i in range(x.shape[1])) for b in np.flatnonzero(~good)):
            return True
    return False
def anonymous(x):
    import hashlib
    return hashlib.sha256(x.encode()).hexdigest()[:16]
def packed(x):
    return x.astype(int)@(1<<np.arange(x.shape[1]))
def boot(arrays):
    out=[];rng=np.random.default_rng(2922)
    for target in sorted(set(k[1] for k in arrays)):
        shared=None
        for (s,t),(rec,hits) in arrays.items():
            if t!=target:continue
            names=sorted(set(rec))
            if shared is None:
                shared=(names,rng.integers(len(names),size=(4000,len(names))))
            names0,draw=shared;assert names==names0
            sums=np.stack([hits[rec==n].sum(0) for n in names])
            size=np.array([(rec==n).sum() for n in names])
            out.append(sums[draw].sum(1)/size[draw].sum(1)[:,None])
    vals=np.mean(out,axis=0)
    return np.quantile(100*vals,[.025,.975],axis=0)
def evaluate():
    pin=WORK/'SEALED.json';seal=read(pin);assert not seal['labels_read'] and seal['code']==sha(HERE/'capture.py')
    assert seal['protocol']==sha(HERE/'PROTOCOL.json') and seal['obs']==sha(ORIGINAL/'SEALED.json')
    metadata=ordinary();by=defaultdict(list);base_tot={.25:2719,.5:1076}
    summary=[];attribution=[];paired=[];rows=[];repro=0.;certchecks=0
    for key,z in bound_components():by[key].append(z)
    for t in (.25,.5):
        arrivals={}
        for direction,parts in by.items():
            src,tgt=direction.split('_to_');f=WORK/(direction+'.npz');assert sha(f)==seal['files'][direction]
            with np.load(f) as d:picks=d['picks'];ids=d['ordinals']
            z={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
            np.testing.assert_equal(ids,z['ordinals'])
            q=support(z);x=np.stack([z[k] for k in ('prior','unary','rel')],axis=-1).astype(float)
            repro=max(repro,float(abs(x.sum(2)-z['final']).max()))
            p=ORIGINAL/('scored_'+direction+'.npz');s=read(p.with_suffix('.json'))
            assert sha(p)==s['file_sha256'] and s['prediction_seal']==seal['obs']
            with np.load(p) as io:pool=io['pool_iou']
            iou=np.take_along_axis(pool,z['slots'].astype(int),1);passed=iou>t
            output=np.take_along_axis(iou,picks.astype(int),1);hit=output>t;v1=hit[:,0]
            missed=~v1&passed.any(1)
            before=has_undominated(x,passed);dom=missed&~before
            escape=np.stack([has_undominated(np.concatenate([x,q[:,p,:]],axis=2),passed)&dom for p in PERMS],axis=1)
            for i in np.flatnonzero(dom)[::61]:
                assert not independent_check(x[i],passed[i])
                for k,perm in enumerate(PERMS):
                    assert independent_check(np.column_stack([x[i],q[i,list(perm),:]]),passed[i])==escape[i,k]
                    certchecks+=1
            rect=np.array([anonymous(tgt+'|'+str(r['recording'])) for r in metadata[tgt]])
            assert len(rect)==len(picks)
            d1=hit[:,1].astype(float)-v1;d2=hit[:,1].astype(float)-hit[:,2:].mean(1)
            arrivals[src,tgt]=(rect,np.column_stack([d1,d2]))
            subsets={'all':np.ones(len(ids),bool),'recoverable_misses':missed,'dominated_misses':dom}
            for label,mask in subsets.items():
                for j,name in enumerate(NAMES):
                    v=dict(direction=direction,threshold=t,scope=label,arm=name,records=int(mask.sum()),hits=int((hit[:,j]&mask).sum()),
                      rescues=int((~v1&hit[:,j]&mask).sum()),harms=int((v1&~hit[:,j]&mask).sum()),
                      different_slots=int(((picks[:,j]!=picks[:,0])&mask).sum()))
                    summary.append(v)
            for k,perm in enumerate(PERMS):
                attribution.append(dict(direction=direction,threshold=t,arm=NAMES[k+1],recoverable=int(missed.sum()),dominated=int(dom.sum()),
                    escape=int(escape[:,k].sum()),rescues_on_escaped=int((escape[:,k]&hit[:,k+1]).sum())))
            for i,idx in enumerate(ids):
                rid=anonymous(f'v1-bound-audit-20261120|ordinary|{direction}|{idx}')
                rows.append(dict(direction=direction,case=rid,recording=rect[i],threshold=t,
                    v1_iou=output[i,0],physical_iou=output[i,1],null_hit_bits=int(packed(hit[i:i+1,2:])[0]),
                    recoverable_miss=int(missed[i]),three_score_dominated=int(dom[i]),physical_escape=int(escape[i,0]),
                    null_escape_bits=int(packed(escape[i:i+1,1:])[0])))
        ci=boot(arrivals)
        delta=100*np.mean([v[1].mean(0) for v in arrivals.values()],axis=0)
        for i,comp in enumerate(('v1','mean_derangements')):
            paired.append(dict(threshold=t,contrast='true_minus_'+comp,delta_macro_pp=delta[i],lo=ci[0,i],hi=ci[1,i]))
        assert sum(r['dominated'] for r in attribution if r['arm']=='true' and r['threshold']==t)==base_tot[t]
    g=pd.DataFrame(summary);g['acc']=100*g.hits/g.records
    g.to_csv(OUT/'direction.csv',index=False)
    avg=g.groupby(['threshold','scope','arm']);table=avg[['hits','rescues','harms','different_slots','records']].sum()
    table['acc_six_equal']=avg.acc.mean();table.to_csv(OUT/'macro.csv')
    pd.DataFrame(attribution).to_csv(OUT/'dominance.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False)
    with gzip.open(OUT/'all_rows.csv.gz','wt',newline='') as dest:
        writer=csv.DictWriter(dest,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len(rows)==37240 and repro<2e-6
    save(OUT/'audit.json',dict(protocol=sha(HERE/'PROTOCOL.json'),seal=sha(pin),records=len(rows)//2,source_steps=0,target_steps=0,
             paired_inputs=len(rows),controls=len(PERMS)-1,scalar_dominance_checks=certchecks,
             reconstructed_score_error=repro,core_weights_changed=False,FINAL_access=False))
if __name__=='__main__':evaluate()
