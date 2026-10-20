"""GT-free risk-weighted boundary coverage and cached-token reveals.

This is an acquisition proxy, not calibrated expected RGB information gain.
Local observations do not actually return binary event-membership labels.
"""
import hashlib
import numpy as np

def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def uniform_order(n=65):
    g=np.rint(np.linspace(0,n-1,8)).astype(int).tolist()
    return g[::2]+g[1::2]+[i for i in range(n)if i not in g]

def reflected(q):
    v=np.ascontiguousarray(q[::-1,::-1].T)
    assert not np.tril(v).any()and abs(v.sum()-1)<1e-12
    return v

def choose(ids,q,selected,count=4):
    """Only physical timestamps, current posterior and already selected IDs."""
    from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
    ids=np.asarray(ids,dtype=np.int64);q=np.asarray(q,dtype=np.float64);n=len(ids)
    assert q.shape==(n,n)and (q>=0).all()and abs(q.sum()-1)<1e-12 and not np.tril(q).any()
    a,b=np.triu_indices(n,1);starts=ids[a];ends=ids[b]+1;iv=final_mbr(ids,q)['interval']
    inter=np.maximum(0,np.minimum(ends,iv[1])-np.maximum(starts,iv[0]));union=ends-starts+iv[1]-iv[0]-inter
    weight=q[a,b]*(1-inter/union)
    # Same min/max physical support as the cached8-frame stride2 local clips.
    lo=np.maximum(ids[0],ids-6);hi=np.minimum(ids[-1],ids+8)+1
    cover_s=(starts[None]>=lo[:,None])&(starts[None]<hi[:,None])
    cover_e=(ends[None]>lo[:,None])&(ends[None]<=hi[:,None])
    chosen=list(selected);trace=[];fallback=uniform_order(n)
    for _ in range(count):
        cs=np.any(cover_s[chosen],axis=0)if chosen else np.zeros(len(a),bool)
        ce=np.any(cover_e[chosen],axis=0)if chosen else np.zeros(len(a),bool)
        gains=(cover_s&~cs[None])@weight+(cover_e&~ce[None])@weight
        eligible=np.ones(n,bool);eligible[chosen]=False;best=float(gains[eligible].max())
        if best>0:
            j=int(np.flatnonzero(eligible&(gains==best))[0]);why='positive_risk_weighted_new_boundary_coverage'
        else:j=next(j for j in fallback if eligible[j]);why='zero_marginal_fallback_uniform_order'
        trace.append(dict(position=j,physical_frame=int(ids[j]),gain=float(gains[j]),all_candidate_gains=gains,
                          eligible=eligible,reason=why))
        chosen.append(j)
    return chosen,dict(state_q_sha256=array_sha(q),state_interval=iv,risk=float(weight.sum()),trace=trace,
        score_definition='posterior*(1-tIoU(candidate,currentMBR)), new start/end mass covered by dense clip hull; not actual observation value')

def reveal(native,positions,values):
    """Only the selected local values enter; unobserved local values absent."""
    import torch
    assert len(positions)==len(set(positions))and len(values)==len(positions)
    out=native.clone();out[positions]=values.to(out);mask=torch.ones(len(native),dtype=torch.bool);mask[positions]=False
    assert torch.equal(out[mask],native[mask])and torch.equal(out[positions],values.to(out))
    return out

def properties():
    ids=np.arange(65)*8;q=np.zeros((65,65));q[10,35]=.6;q[12,38]=.4
    selected,meta=choose(ids,q,[],4);assert len(set(selected))==4
    s2,_=choose(ids,q,selected,4);assert len(set(s2))==8
    qr=reflected(q);assert np.array_equal(reflected(qr),q)
    z=np.zeros((65,65));z[10,35]=1.;zz,m=choose(ids,z,[],4);assert zz==uniform_order()[:4]and m['risk']==0
    import torch
    n=torch.arange(65*2,dtype=torch.float32).reshape(65,2);v=torch.zeros(4,2);out=reveal(n,selected,v)
    assert not torch.equal(out,n)and torch.equal(reveal(n,[],n[:0]),n)
    # Nothing but the selected values is an input to reveal; selector has no feature input.
    return dict(status='pass',unique_budget=True,point_state_fallback=True,reflected_state_involution=True,
                zero_reveal_exact=True,unobserved_feature_argument_absent=True,GT_or_model_run=False)
