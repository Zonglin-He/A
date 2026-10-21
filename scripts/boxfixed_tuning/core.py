"""Parameterized output-only BoxFixed inference. Heavy-network inputs never change."""
import time
import numpy as np
import torch
from scipy.special import logsumexp
from stvg.temporal import probabilities,mass_table,final_mbr
from stvg.spatial import evidence_dp
from scripts.tube_fusion.core import restore_geometry
BASE=dict(temperature=1.,observed_weight=.5,guard_weight=.1,link_weight=.25)

def check_params(p):
    assert set(p)==set(BASE)
    assert p['temperature']>0 and 0<=p['observed_weight']<=1
    assert p['guard_weight']>=0 and p['link_weight']>=0
    assert all(np.isfinite(v)for v in p.values())

def temporal(cap,temperature):
    tables=[];begin=time.perf_counter()
    for layer in range(6):
        z=[a[layer]for a in cap['logits']];grids=probabilities(cap['records'],z)
        if temperature!=1.:
            for g,a in zip(grids,z):
                a=np.asarray(a)[0].astype(np.float64);ij=g['ij']
                v=(a[ij[0],0]+a[ij[1],1])/temperature
                g['p']=np.exp(v-logsumexp(v))
        ids,q=mass_table(grids);assert ids==list(cap['frame_ids']);tables.append(q)
    q=np.mean(tables,axis=0);interval=final_mbr(ids,q)['interval']
    positive=q[q>0];fps=cap.get('fps',1.)
    lengths=(np.asarray(ids)[None]+1-np.asarray(ids)[:,None])/fps
    return dict(interval=interval,q=q,entropy=float(-(positive*np.log(positive)).sum()),
        expected_length_seconds=float((q*lengths).sum()),seconds=time.perf_counter()-begin)

def spatial(cap,interval,p):
    check_params(p);g=restore_geometry(cap['geometry']);bank=g.for_interval(interval)
    if any(p[k]!=BASE[k]for k in ('observed_weight','guard_weight','link_weight')):
        obs=bank.event & bank.observed;unobs=bank.event & ~bank.observed
        ow=p['observed_weight']*obs.double()/max(1,int(obs.sum()))
        uw=(1-p['observed_weight'])*bank.confidence*unobs/max(1,int(unobs.sum()))
        bank.score_unary=(ow+uw)[:,None]*bank.projection-p['guard_weight']*bank.guard/bank.frames
        ew=p['link_weight']*bank.event_edges*bank.confidence[:-1]*bank.confidence[1:]/max(1,int(bank.event_edges.sum()))
        bank.score_pair=ew[:,None,None]*bank.link
    out=evidence_dp(bank)
    return out
